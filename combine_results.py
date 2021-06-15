import geopandas as gpd
import geojson
import rasterio
import math
import numpy as np
from plyfile import PlyData
from shapely.geometry import Point
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio import mask
from rasterio.merge import merge
from shapely.geometry import box
from scipy.spatial import Delaunay
import scipy

#Combine restult of geometry obstacle detection and image classification
def combine_rasters(pc,pct):
	for index, rows in pc.iterrows():
		with rasterio.open('./raster_results/'+rows.identificatie+".tif") as src:
			r = src.read(1)
			pcr, out_transform = rasterio.mask.mask(src, pct.geometry)
		pcr[0] = np.where(pcr[0]> 0, 1, pcr[0])
		final = np.copy( pcr)
		for (x,y), value in np.ndenumerate(pcr[0]): 
			if pcr[0][x,y]==1 and r[x,y]==2: #Good Pixel
				final[0][x,y]=100
			elif pcr[0][x,y]==0 and r[x,y]==2: #Medium Pixel
				final[0][x,y]=50
			elif r[x,y]==1 and pcr[0][x,y]==1: #Medium Pixel
				final[0][x,y]=50
			elif r[x,y]==1 and pcr[0][x,y]==0: #Bad pixel
			    final[0][x,y]=0
			else:
				final[0][x,y]=-9999 #non existant pixel
		new_array = rasterio.open('./combine_results/'+rows.identificatie+'_mix.tif',
                      'w', driver='GTiff',
                         height =pcr.shape[1], width = pcr.shape[2],
                          count=1, dtype = str(pcr[0][0].dtype),
                          crs=28992, nodata = -9999,
                          transform=out_transform)
		new_array.write(final[0],1)
		new_array.close()	

#Get normal of a point cloud
def get_normal(nx,ny,nz):
    z1 = 0
    z2 = 0
    z3 = 1
    # angle between two vectors in radians
    angle = np.arccos([((z1 * nx) + (z2 * ny) + (z3 * nz))
                       / (math.sqrt((z1 * z1) + (z2 * z2) + (z3 * z3)) 
                       * math.sqrt((nx * nx) + (ny * ny) + (nz * nz)))])
    return (math.degrees(angle))

#IDW interpolation for a point cloud
def idw_interpolation(list_pts_3d, identificatie):
    #get the extent of the points
    xmin = list_pts_3d.transpose()[0].min()-2
    xmax = list_pts_3d.transpose()[0].max()+2
    ymin = list_pts_3d.transpose()[1].min()-2
    ymax = list_pts_3d.transpose()[1].max()+2
    #Create another list with only the x,y coordinate
    list_pts_2d = []
    for j in list_pts_3d:
        list_pts_2d.append([j[0],j[1]])
    #Make a tree with the 2d list coordinates
    kd = scipy.spatial.KDTree(list_pts_2d)
    # Get the max extent in order to have all points in the raster
    cell = 0.25
    def get_max_raster(maxi, mini, cell):
        if (maxi-mini)%cell == 0:
            res = maxi
        else:
            res= maxi + (cell - (maxi-mini)%cell)
        return res
    xmax = get_max_raster(xmax, xmin, cell)
    ymax = get_max_raster(ymax, ymin, cell)
    # Get the centroids of the extents
    def centroid(coor,cell):
        return coor + cell/2.0
    cenxmin = centroid(xmin,cell)
    cenymin = centroid(ymin,cell)
    cenxmax = centroid(xmax,cell)
    cenymax = centroid(ymax,cell)
    # make a grid with the centroids and perform the query to get the minimun distance and the nearest neighbor.
    x, y = np.mgrid[cenxmin:cenxmax:cell, cenymin:cenymax:cell]
    pts = list(zip(x.ravel(), y.ravel()))
    radius = 5
    power = 2
    d,i = kd.query(pts, radius)
    # Euclidian distance between two points
    def distance(x1,y1,x2,y2):
        d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return d
    # Loop to traverse the point and make a list with the heights, distance and weight from pts to 3d points
    ll = []
    for h, row in enumerate(pts):
        l = []
        for points_aux in i[h]:
            dist =distance(list_pts_3d[points_aux][0],list_pts_3d[points_aux][1], row[0],row[1])
            weight = math.pow(dist,-power) if dist>0 else 10000
            l.append([list_pts_3d[points_aux][2],dist,weight])
        ll.append(l)
    # Loop to traverse the list made before with the total weight of each point of the grid
    total_w = []
    for w in ll:
        total_weight = 0
        for weights in w:
            total_weight += weights[2]
        total_w.append(total_weight)
    # loop to interpolate the grid with the weights
    idw = []
    for r, h in enumerate(ll):
        if len(h) == 0:
            z=-9999
        else:
            z = 0
            q = 0
            boolean = True
            while boolean == True and q<len(h):
                if h[q][1] == 0:
                    z = h[q][0]
                    boolean = False
                else:
                    z += (h[q][0]*h[q][2])/total_w[r]
                    q = q+ 1
        idw.append(z)
    #Delaunay Triangulation to get the convex hull
    hull =  scipy.spatial.Delaunay(list_pts_2d)
    # Creation of the final list with x,y,z interpolated to be saved in ASCI
    final = []
    a = 0 
    while a< len(idw):
        final.append([pts[a][0],pts[a][1],idw[a]])
        a = a +1
    #Sort the created list to be readable to asc
    final.sort(key = lambda x: x[1], reverse= True) 
    #Create the asc format
    with open("./normal_rasters/"+identificatie + '.ascii', 'w') as asc:
        s = '''ncols {0} 
nrows {1} 
xllcorner {2}
yllcorner {3}
cellsize {4}
nodata_value -9999 
'''.format(len(x),len(y[0]), xmin, ymin,cell)
        asc.write(s)
        for w,t in enumerate(final):
            h = '{0} '.format(round(t[2],4))
            if w%(len(x)) == 0 and w>0:
                asc.write('\n')
            asc.write(h)

    #print("File"+ identificatie +  "written",)

    return len(x)*len(y[0])*cell


def create_normal_raster(pc,data_pd):
	inte = []
	for index, rows in pc.iterrows():
		df1 = gpd.GeoDataFrame({'geometry': rows.geometry, 'df1': [index]})
		df1.crs = 28992
		intersections = gpd.overlay(data_pd,df1, how='intersection')
		normals = []
		idtie = []
		for index1, rows1 in intersections.iterrows():
			norm = get_normal(rows1.nx, rows1.ny, rows1.nz)   
			normals.append(norm)
			idtie.append(rows.identificatie)
		intersections['normals']= normals
		intersections['identificatie']= idtie
		pci = intersections.loc[intersections.normals<50]
		inte.append(pci)
	for i in inte:
		points = i.to_numpy()
		ident = points[:,10][0]
		points = points[:, (0, 1,9)]
		idw_interpolation(points,ident)
 
 #Compute the are of the obstacles
def compute_area(pc):
	area = []
	for index, rows in pc.iterrows():
		with rasterio.open('./normal_rasters/'+rows.identificatie + '.ascii') as f:
			with rasterio.open('./combine_results/'+rows.identificatie+'_mix.tif') as e:
				fe = f.read(1)
				px, py = e.index(f.bounds[0], f.bounds[3])
				ee = e.read(1,  window=Window( py,px, f.shape[1], f.shape[0]))
		final = np.copy( fe)
		for (y,x), value in np.ndenumerate(ee):
			if ee[y,x]==50:
				final[y,x]=0.5 * 0.25 * 0.25* (1/math.cos(math.radians(fe[y,x])))
			elif ee[y,x]==10:
				final[y,x]=0.25 * 0.25* (1/math.cos(math.radians(fe[y,x])))
			else:
				final[y,x]=0
		area.append((rows.identificatie,final.sum()))
	return area 


def main(obst_geo):
	pc = gpd.read_file('./data/classification_features/data.geojson')
	pct = gpd.read_file(obst_geo)
	combine_rasters(pc,pct)
	# -- READ PLY: store the input 3D points in np array
	plydata = PlyData.read('./data/brink/50_bdg_rad5.ply')        # read file
	data = plydata.elements[0].data                         # read data
	df = gpd.GeoDataFrame(data)
	data_pd = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y, df.z)) # Convert to DataFrame, because DataFrame can parse structured data
	data_pd.crs = 28992
	create_normal_raster(pc,data_pd)
	area = compute_area(pc)
	colnames =['identificatie','obstacle_area']
	df = gpd.GeoDataFrame(area, columns = colnames)
	df.to_csv('roof_semantics.csv',index=False)
	print("final results ready")
if __name__ == '__main__':
    main(obst_geo)