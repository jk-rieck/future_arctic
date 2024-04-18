import numpy as np
from sklearn.neighbors import BallTree
from skimage.morphology import skeletonize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

def extract_transect(lon_l,lat_l,dataset,n=None):
#     if n==None:
#         n=np.ones(len(lon_l))*400
    
#     if len(lon_l) != len(lat_l):
#         raise ValueError('lon_l and lat_l must have the same lenght')
    
    coords = np.vstack([lat_l,lon_l]).T
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
    
    
def create_extra_axis(ax,ds):
    ax1_t = ax.twiny()
    ax1_t.spines["bottom"].set_position(("axes", -0.25))
    ax1_t.set_frame_on(True)
    ax1_t.patch.set_visible(False)
    ax1_t.xaxis.set_ticks_position("bottom")
    ax1_t.xaxis.set_label_position("bottom")

    ax1_t.plot(ds.diag*0,alpha=0)
    
    return ax1_t

def extract_transects(transects_coords, ds_ref, ds_fut , var=None, outfile="output_{0}.nc"):
    transects_data = []
    for counter, transect in enumerate(transects_coords):
        # print(counter,transect[0])
        
        x = transect[0]
        y = transect[1]
        if len(x) != len(y):
            raise ValueError("Lenght of transect x and y should be identical")
            
        index_y ,index_x = extract_transect(x,y,ds_ref)
        
        ds_xi = xr.DataArray(index_x.ravel(), dims=["x_points"])
        ds_yi = xr.DataArray(index_y.ravel(), dims=["y_points"])
        
        ds_REF = ds_ref[var].assign_coords({"xt{0}".format(counter):ds_ref.x,"yt{0}".format(counter):ds_ref.y}).to_dataset().swap_dims({"x":"xt{0}".format(counter),"y":"yt{0}".format(counter)})
        transect_REF = ds_REF.isel({"xt{0}".format(counter):ds_xi,"yt{0}".format(counter):ds_yi}).rename({var:var+"_ref_t{0}".format(counter)})
        transect_REF = transect_REF.rename({"x_points":"x_points"+"_t{0}".format(counter),"y_points":"y_points"+"_t{0}".format(counter)})

        ds_FUT = ds_fut[var].assign_coords({"xt{0}".format(counter):ds_fut.x,"yt{0}".format(counter):ds_fut.y}).to_dataset().swap_dims({"x":"xt{0}".format(counter),"y":"yt{0}".format(counter)})
        transect_FUT = ds_FUT.isel({"xt{0}".format(counter):ds_xi,"yt{0}".format(counter):ds_yi}).rename({var:var+"_fut_t{0}".format(counter)})
        transect_FUT = transect_FUT.rename({"x_points":"x_points"+"_t{0}".format(counter),"y_points":"y_points"+"_t{0}".format(counter)})

        # transect_FUT = ds_fut[var].assign_coords({"xt{0}".format(counter):ds_fut.x,"yt{0}".format(counter):ds_fut.y})
        # transect_REF.isel({"xt{0}".format(counter):ds_xi,"yt{0}".format(counter):ds_yi}).rename(var+"_fut_t{0}".format(counter))
        
        diag = xr.DataArray(np.arange(len(x)), dims="diag")
        diag_transect_REF = transect_REF.isel({"x_points"+"_t{0}".format(counter):diag, "y_points"+"_t{0}".format(counter):diag})
        diag_transect_FUT = transect_FUT.isel({"x_points"+"_t{0}".format(counter):diag, "y_points"+"_t{0}".format(counter):diag})
        
        transects_data.append([diag_transect_REF, diag_transect_FUT])
    
    transect_dataset = { "transect_{0}".format(ii) : xr.merge(transects_data[ii]) for ii in range(len(transects_data))}
    
    print("Storing data")
    
    [item.to_netcdf(outfile.format(key)) for key,item in transect_dataset.items()]
    
    return transect_dataset

def create_extra_axis(ax,ds,position=-0.25):
    ax1_t = ax.twiny()
    ax1_t.spines["bottom"].set_position(("axes", position))
    ax1_t.set_frame_on(True)
    ax1_t.patch.set_visible(False)
    ax1_t.xaxis.set_ticks_position("bottom")
    ax1_t.xaxis.set_label_position("bottom")

    ax1_t.plot(ds.diag*0,alpha=0)
    
    ax1_t.set_xlim(0,max(ds.diag))
    ax1_t.set_xticks(np.arange(0,max(ds.diag)-1,100))
    
    return ax1_t


def latitude_labels(ax,ds):
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [np.round(ds.nav_lat.isel(diag=int(label)).values) for label in labels]

    ax.set_xticklabels(labels)
    # ax.set_xlabel('Latitude')
    
    
def plot_transects(ds=None,ds_cf=None,ds_ice=None,contour=True, contourf=False, varname="",ylim=(0,500),figsize=(7,7),output="out.png", kwargs_plot=None):
    fig = plt.figure(figsize=(7, 7))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1_1 = fig.add_axes([1, 0.95, 0.65, 0.2])
    ax1_2 = fig.add_axes([1, 0.6, 0.65, 0.2])
    
    ax2 = fig.add_subplot(3, 1, 3)
    ax2_1 = fig.add_axes([1, 0.22, 0.65, 0.2])
    ax2_2 = fig.add_axes([1, -0.13, 0.65, 0.2])
    
    axis = [ax1,ax1_1,ax1_2,ax2,ax2_1,ax2_2]

    ax1.set_title("REF")
    ax2.set_title("FUT")
    
    if ds and not contourf:
        data2plot_REF_t0 = ds['REF']["t0"]
        data2plot_REF_t1 = ds['REF']["t1"]
        data2plot_REF_t2 = ds['REF']["t2"]
        data2plot_FUT_t0 = ds['FUT']["t0"]
        data2plot_FUT_t1 = ds['FUT']["t1"]
        data2plot_FUT_t2 = ds['FUT']["t2"]
        
        cbar = data2plot_REF_t0.plot(x='diag', ax=ax1, **kwargs_plot, rasterized=True)
        data2plot_REF_t1.plot(x='diag', ax=ax1_1, **kwargs_plot, rasterized=True)
        data2plot_REF_t2.plot(x='diag', ax=ax1_2, **kwargs_plot, rasterized=True)
        data2plot_FUT_t0.plot(x='diag', ax=ax2, **kwargs_plot, rasterized=True)
        data2plot_FUT_t1.plot(x='diag', ax=ax2_1, **kwargs_plot, rasterized=True)
        data2plot_FUT_t2.plot(x='diag', ax=ax2_2, **kwargs_plot, rasterized=True)
        
        cax = plt.axes((0.15, -0.05, 0.7, 0.03))
        cbar = plt.colorbar(cbar, cax=cax,orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        
        cbar.set_label(varname)
    
    elif ds and contourf: 
        data2plot_REF_t0 = ds['REF']["t0"]
        data2plot_REF_t1 = ds['REF']["t1"]
        data2plot_REF_t2 = ds['REF']["t2"]
        data2plot_FUT_t0 = ds['FUT']["t0"]
        data2plot_FUT_t1 = ds['FUT']["t1"]
        data2plot_FUT_t2 = ds['FUT']["t2"]
        
        cbar = data2plot_REF_t0.plot.contourf(x='diag', ax=ax1, **kwargs_plot)
        data2plot_REF_t1.plot.contourf(x='diag', ax=ax1_1, **kwargs_plot)
        data2plot_REF_t2.plot.contourf(x='diag', ax=ax1_2, **kwargs_plot)
        data2plot_FUT_t0.plot.contourf(x='diag', ax=ax2, **kwargs_plot)
        data2plot_FUT_t1.plot.contourf(x='diag', ax=ax2_1, **kwargs_plot)
        data2plot_FUT_t2.plot.contourf(x='diag', ax=ax2_2, **kwargs_plot)
        
        cax = plt.axes((0.15, -0.05, 0.7, 0.03))
        cbar = plt.colorbar(cbar, cax=cax,orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        
        cbar.set_label(varname)
    
    if contour and ds_cf:
        # Reference run
        data2plot_REF_rho_t0 = ds_cf['REF']["t0"]
        data2plot_REF_rho_t1 = ds_cf['REF']["t1"]
        data2plot_REF_rho_t2 = ds_cf['REF']["t2"]
        data2plot_FUT_rho_t0 = ds_cf['FUT']["t0"]
        data2plot_FUT_rho_t1 = ds_cf['FUT']["t1"]
        data2plot_FUT_rho_t2 = ds_cf['FUT']["t2"]
        
        cs = data2plot_REF_rho_t0.where(data2plot_REF_rho_t0> 1000).plot.contour(x='diag', ax=ax1,vmin=1020,vmax=1030,cmap=cmo.dense,levels=[1024,1026,1027])
        ax1.clabel(cs, inline=1, fontsize=10)

        cs = data2plot_REF_rho_t1.where(data2plot_REF_rho_t1> 1000).plot.contour(x='diag', ax=ax1_1,vmin=1020,vmax=1030,cmap=cmo.dense,levels=[1024,1026,1027])
        ax1_1.clabel(cs, inline=1, fontsize=10)
    
        cs = data2plot_REF_rho_t2.where(data2plot_REF_rho_t2> 1000).plot.contour(x='diag', ax=ax1_2,vmin=1020,vmax=1030,cmap=cmo.dense,levels=[1024,1026,1027])
        ax1_2.clabel(cs, inline=1, fontsize=10)

        # Future run
        cs = data2plot_FUT_rho_t0.where(data2plot_FUT_rho_t0> 1000).plot.contour(x='diag',ax=ax2,vmin=1020,vmax=1030,cmap=cmo.dense,levels=[1024,1026,1027])
        ax2.clabel(cs, inline=1, fontsize=10)

        cs = data2plot_FUT_rho_t1.where(data2plot_FUT_rho_t1> 1000).plot.contour(x='diag',ax=ax2_1,vmin=1020,vmax=1030,cmap=cmo.dense,levels=[1024,1026,1027])
        ax2_1.clabel(cs, inline=1, fontsize=10)

        cs = data2plot_FUT_rho_t2.where(data2plot_FUT_rho_t2> 1000).plot.contour(x='diag',ax=ax2_2,vmin=1020,vmax=1030,cmap=cmo.dense,levels=[1024,1026,1027])
        ax2_2.clabel(cs, inline=1, fontsize=10)

    if ds_ice:
        counter=0
        for key, items in ds_ice.items():
            for key1,item in items.items():
                data2plot = item
                if counter==3:
                    fix_pos= -0.043
                else:
                    fix_pos=0
                
                ice_ax = fig.add_axes([axis[counter].get_position().x0,axis[counter].get_position().y1+fix_pos,axis[counter].get_position().width,0.05])
                ice_ax.fill_between(data2plot.diag,data2plot*0,data2plot,alpha=0.5)
                ice_ax.grid()
                ice_ax.set_ylim((0,3))
                ice_ax.set_xlim((0,max(data2plot.diag)))
                ice_ax.xaxis.set_ticklabels([])
                counter+=1
    
    # Style of plots    
    [ax.set_ylim(*ylim) for ax in axis]

    [ax.invert_yaxis() for ax in axis]

    [ax.set_xlabel('') for ax in axis]

    [ax.set_ylabel('Depth [m]') for ax in axis]

    ax1_t = create_extra_axis(ax1,data2plot_REF_rho_t0)
    ax2_t = create_extra_axis(ax2,data2plot_FUT_rho_t0)

    ax1_1_t = create_extra_axis(ax1_1,data2plot_REF_rho_t1)
    ax1_2_t = create_extra_axis(ax1_2,data2plot_FUT_rho_t2)

    ax2_1_t = create_extra_axis(ax2_1,data2plot_REF_rho_t1)
    ax2_2_t = create_extra_axis(ax2_2,data2plot_FUT_rho_t2)
    
    
    fig.canvas.draw()

    axis_t = [ax1_t,ax1_1_t,ax1_2_t,ax2_t,ax2_1_t,ax2_2_t]

    latitude_labels(ax1_t,data2plot_REF_rho_t0)
    latitude_labels(ax2_t,data2plot_REF_rho_t0)

    latitude_labels(ax1_1_t,data2plot_REF_rho_t1)
    latitude_labels(ax1_2_t,data2plot_REF_rho_t2)

    latitude_labels(ax2_1_t,data2plot_FUT_rho_t1)
    latitude_labels(ax2_2_t,data2plot_FUT_rho_t2)
    
    t = fig.text(0.4, 0.95, 'REF simulation', fontsize=16, weight=1000, va='center')
    t = fig.text(0.4, 0.36, 'FUT simulation', fontsize=16, weight=1000, va='center')
    
    plt.subplots_adjust(hspace=0.6)
    plt.savefig(output, bbox_inches='tight')
    
def plot_transects_diff(ds=None,ds_cf=None,ds_ice=None,contour=True, contourf=False, varname="",ylim=(0,500),figsize=(7,7),output="out.png", kwargs_plot=None):
    fig = plt.figure(figsize=(7, 7))

    ax1 = fig.add_subplot(3, 1, 2)
    ax1_1 = fig.add_axes([1, 0.6, 0.65, 0.2])
    ax1_2 = fig.add_axes([1, 0.2, 0.65, 0.2])

    axis = [ax1,ax1_1,ax1_2]
    
    if ds and not contourf:
        data2plot_REF_t0 = ds['REF']["t0"]
        data2plot_REF_t1 = ds['REF']["t1"]
        data2plot_REF_t2 = ds['REF']["t2"]
        data2plot_FUT_t0 = ds['FUT']["t0"]
        data2plot_FUT_t1 = ds['FUT']["t1"]
        data2plot_FUT_t2 = ds['FUT']["t2"]
        
        cbar = (data2plot_FUT_t0 - data2plot_REF_t0).plot(x='diag', ax=ax1, **kwargs_plot, rasterized=True)
        (data2plot_FUT_t1 - data2plot_REF_t1).plot(x='diag', ax=ax1_1, **kwargs_plot, rasterized=True)
        (data2plot_FUT_t2 - data2plot_REF_t2).plot(x='diag', ax=ax1_2, **kwargs_plot, rasterized=True)
                
        cax = plt.axes((0.15, 0.25, 0.7, 0.03))
        cbar = plt.colorbar(cbar, cax=cax,orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.xaxis.get_major_locator().numticks = 5
        
        cbar.set_label(varname)
    
    elif ds and contourf: 
        data2plot_REF_t0 = ds['REF']["t0"]
        data2plot_REF_t1 = ds['REF']["t1"]
        data2plot_REF_t2 = ds['REF']["t2"]
        data2plot_FUT_t0 = ds['FUT']["t0"]
        data2plot_FUT_t1 = ds['FUT']["t1"]
        data2plot_FUT_t2 = ds['FUT']["t2"]
        
        cbar = (data2plot_FUT_t0 - data2plot_REF_t0).plot.contourf(x='diag', ax=ax1, **kwargs_plot)
        (data2plot_FUT_t1 - data2plot_REF_t1).plot.contourf(x='diag', ax=ax1_1, **kwargs_plot)
        (data2plot_FUT_t2 - data2plot_REF_t2).plot.contourf(x='diag', ax=ax1_2, **kwargs_plot)
        
        cax = plt.axes((0.15, 0.25, 0.7, 0.03))
        cbar = plt.colorbar(cbar, cax=cax,orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.xaxis.get_major_locator().numticks = 5
        
        cbar.set_label(varname)
    
    
    # Style of plots    
    [ax.set_ylim(*ylim) for ax in axis]

    [ax.invert_yaxis() for ax in axis]

    [ax.set_xlabel('') for ax in axis]

    [ax.set_ylabel('Depth [m]') for ax in axis]

    ax1_t = create_extra_axis(ax1,data2plot_REF_t0)
    ax1_1_t = create_extra_axis(ax1_1,data2plot_REF_t1)
    ax1_2_t = create_extra_axis(ax1_2,data2plot_REF_t2)

    fig.canvas.draw()

    axis_t = [ax1_t,ax1_1_t,ax1_2_t]

    latitude_labels(ax1_t,data2plot_REF_t0)
    latitude_labels(ax1_1_t,data2plot_REF_t1)
    latitude_labels(ax1_2_t,data2plot_REF_t2)
    
    t = fig.text(0.4, 0.65, 'DIFF (FUT-REF)', fontsize=16, weight=1000, va='center')
    
    plt.subplots_adjust(hspace=0.6)
    plt.savefig(output, bbox_inches='tight')
    