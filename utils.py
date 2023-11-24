import numpy as np
from sklearn.neighbors import BallTree
from skimage.morphology import skeletonize

def extract_transect(lon_l,lat_l,dataset,n=None):
#     if n==None:
#         n=np.ones(len(lon_l))*400
    
#     if len(lon_l) != len(lat_l):
#         raise ValueError('lon_l and lat_l must have the same lenght')

    x_0 = np.linspace(-150,-150,400)
    y_0 = np.linspace(70,90,400)

    x_1 = np.linspace(-10,-10,300)
    y_1 = np.linspace(90,75,300)

    x_i = np.hstack((x_0,x_1))
    y_i = np.hstack((y_0,y_1))
    
    coords = np.vstack([y_i,x_i]).T
    grid_coords = np.vstack([dataset.nav_lat.values.flat, dataset.nav_lon.values.flat]).T
    
    ball_tree = BallTree(np.deg2rad(grid_coords), metric='haversine')
    
    distances_radians, _ = ball_tree.query(np.deg2rad(coords), return_distance=True,breadth_first=True)
    
    index_y ,index_x = np.unravel_index(_, dataset.nav_lat.shape)
    
    return index_y,index_x



def map_config(ax):
    ax.set_extent([-180, 180, 70, 90], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    
    
def create_extra_axis(ax):
    ax1_t = ax.twiny()
    ax1_t.spines["bottom"].set_position(("axes", -0.25))
    ax1_t.set_frame_on(True)
    ax1_t.patch.set_visible(False)
    ax1_t.xaxis.set_ticks_position("bottom")
    ax1_t.xaxis.set_label_position("bottom")

    ax1_t.plot(data2plot_FUT.diag*0,alpha=0)
    ax1_t.set_xticks([0,100,200,300,400,500,600,699])
    
    return ax1_t
