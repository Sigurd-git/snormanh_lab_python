import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def line_plot(result_df,x,y,xlim=None,ylim=None,height=10,aspect=1,facet_col=None,facet_row=None,color_variable=None,style_variable=None,repeat_variable=None,average=False,legend=True,axg=None):
    '''
    result_df: dataframe with columns x,y,color_variable,style_variable,repeat_variable,facet_col,facet_row,where x,y is necessary and others are optional.
    x: the name of the column in result_df that will be used as x
    y: the name of the column in result_df that will be used as y

    xlim: the range of x
    ylim: the range of y

    height: the height of the figure
    aspect: the aspect ratio of the figure

    facet_col: the name of the column in result_df that will be used to devide plot into several columns.
    facet_row: the name of the column in result_df that will be used to devide plot into several rows.

    color_variable: the name of the column in result_df that will be used to group data and give each group a seperate color.
    style_variable: the name of the column in result_df that will be used to group data and give each group a seperate style.
    repeat_variable: the name of the column in result_df that will be used to group data and give each group a seperate line(With same color and style). This is not supported yet.

    axg: the axes of the figure. If None, a new figure will be created.
    '''
    if repeat_variable is not None:
        assert False, 'repeat_variable is not supported yet'
    #facet
    match facet_row,facet_col:
        case (None,None):
            if axg is None:
                _,ax = plt.subplots(1,1,figsize=(height * aspect,height))
            else:
                ax = axg
            
            sns.lineplot(data=result_df,x=x,y=y,hue=color_variable,style=style_variable,ax=ax)
            if average:
                sns.lineplot(data=result_df,x=x,y=y,ax=ax,estimator=np.mean,color='green',linewidth=3)
            ax.set_xlabel(f'{x}')
            ax.set_ylabel(f'{y}')

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if not legend:
                ax.legend_.remove()
            return ax
        case (None,_):
            if axg is None:
                g = sns.FacetGrid(result_df,col=facet_col,row=facet_row,height=10,aspect=aspect)
            else:
                assert len(g.axes_dict) == len(np.unique(result_df[facet_col])), 'axg should have same number of columns as unique values in facet_col'
                g = axg
            for (col_val), ax in g.axes_dict.items():
                tmp = result_df[(result_df[facet_col]==col_val)]
                sns.lineplot(data=tmp,x=x,y=y,hue=color_variable,style=style_variable,ax=ax)
                if average:
                    sns.lineplot(data=tmp,x=x,y=y,ax=ax,estimator=np.mean,color='green',linewidth=3)
                ax.set_xlabel(f'{x}')
                ax.set_ylabel(f'{y}')

                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if not legend:
                    ax.legend_.remove()
            return g
        case (_,None):
            if axg is None:
                g = sns.FacetGrid(result_df,col=facet_col,row=facet_row,height=10,aspect=aspect)
            else:
                assert len(g.axes_dict) == len(np.unique(result_df[facet_row])), 'axg should have same number of rows as unique values in facet_row'
                g = axg
            for (row_val), ax in g.axes_dict.items():
                tmp = result_df[(result_df[facet_row]==row_val)]
                sns.lineplot(data=tmp,x=x,y=y,hue=color_variable,style=style_variable,ax=ax)
                if average:
                    sns.lineplot(data=tmp,x=x,y=y,ax=ax,estimator=np.mean,color='green',linewidth=3)
                ax.set_xlabel(f'{x}')
                ax.set_ylabel(f'{y}')

                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if not legend:
                    ax.legend_.remove()
            return g
        case (_,_):
            if axg is None:
                g = sns.FacetGrid(result_df,col=facet_col,row=facet_row,height=10,aspect=aspect)
            else:
                assert len(g.axes_dict) == len(np.unique(result_df[facet_row])) * len(np.unique(result_df[facet_col])), 'axg should have same number of rows and columns as unique values in facet_row and facet_col'
                g = axg
            for (row_val, col_val), ax in g.axes_dict.items():
                tmp = result_df[(result_df[facet_row]==row_val) & (result_df[facet_col]==col_val)]
                sns.lineplot(data=tmp,x=x,y=y,hue=color_variable,style=style_variable,ax=ax)
                if average:
                    sns.lineplot(data=tmp,x=x,y=y,ax=ax,estimator=np.mean,color='green',linewidth=3)
                ax.set_xlabel(f'{x}')
                ax.set_ylabel(f'{y}')

                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if not legend:
                    ax.legend_.remove()
            return g


def STRF_plot(result_df,STRF,height=10,aspect=1,facet_col=None,facet_row=None,vmin=None,vmax=None):
    '''
    result_df: dataframe with columns x,y,color_variable,style_variable,repeat_variable,facet_col,facet_row,where x,y is necessary and others are optional.
    STRF: the name of the column in result_df that will be used as STRF

    height: the height of the figure
    aspect: the aspect ratio of the figure

    facet_col: the name of the column in result_df that will be used to devide plot into several columns.
    facet_row: the name of the column in result_df that will be used to devide plot into several rows.


    '''

    #facet
    match facet_row,facet_col:
        case (None,None):
            fig,ax = plt.subplots(1,1,figsize=(height * aspect,height))
            STRF_matrix = result_df[STRF].values[0]
            sns.heatmap(STRF_matrix,ax=ax,cmap='RdBu_r',cbar=False,vmin=vmin,vmax=vmax)

            return ax
        case (None,_):
            g = sns.FacetGrid(result_df,col=facet_col,row=facet_row,height=10,aspect=aspect)
            for (col_val), ax in g.axes_dict.items():
                STRF_matrix = result_df[(result_df[facet_col]==col_val)][STRF].values[0]
                sns.heatmap(STRF_matrix,ax=ax,cmap='RdBu_r',cbar=False,vmin=vmin,vmax=vmax)

            return g
        case (_,None):
            g = sns.FacetGrid(result_df,col=facet_col,row=facet_row,height=10,aspect=aspect)
            for (row_val), ax in g.axes_dict.items():
                STRF_matrix = result_df[(result_df[facet_row]==row_val)][STRF].values[0]
                sns.heatmap(STRF_matrix,ax=ax,cmap='RdBu_r',cbar=False,vmin=vmin,vmax=vmax)
            return g
        case (_,_):
            g = sns.FacetGrid(result_df,col=facet_col,row=facet_row,height=10,aspect=aspect)
            for (row_val, col_val), ax in g.axes_dict.items():
                STRF_matrix = result_df[(result_df[facet_row]==row_val) & (result_df[facet_col]==col_val)][STRF].values[0]
                sns.heatmap(STRF_matrix,ax=ax,cmap='RdBu_r',cbar=False,vmin=vmin,vmax=vmax)
            return g



def trf_plot(result_df,facet_col=None,facet_row=None,color_variable=None,style_variable=None,repeat_variable=None):


    #get max lag
    result_df['max_lag'] = result_df['TRF'].apply(lambda x: np.argmax(x))

    #merge
    TRF = np.vstack(result_df['TRF'].values)

    #to result_dfframe
    TRF = pd.DataFrame(TRF)
    
    TRF = pd.concat([result_df,TRF],axis=1)

    #drop columns
    TRF = TRF.drop(columns=['TRF'])

    result_df_columns = result_df.columns
    #drop TRF
    result_df_columns = result_df_columns.drop('TRF')
    #melt
    TRF = TRF.melt(id_vars=result_df_columns,var_name='lag',value_name='TRF')

    if repeat_variable is not None:
        assert False, 'repeat_variable is not supported yet'
    #facet
    match facet_row,facet_col:
        case (None,None):
            fig,axes = plt.subplots(1,1,figsize=(10,10))
            max_lags = result_df['max_lag'].values
            sns.lineplot(data=TRF,x='lag',y='TRF',style=style_variable,hue=color_variable,ax=axes)
            for i,max_lag in enumerate(max_lags):
                axes.axvline(max_lag,color='black')
            return axes
        case (None,_):
            g = sns.FacetGrid(TRF,col=facet_col,row=facet_row,height=10)
            for (col_val), ax in g.axes_dict.items():
                max_lags = result_df[(result_df[facet_col]==col_val)]['max_lag'].values
                tmp = TRF[(TRF[facet_col]==col_val)]
                sns.lineplot(data=tmp,x='lag',y='TRF',hue=color_variable,style=style_variable,ax=ax)
                for i,max_lag in enumerate(max_lags):
                    ax.axvline(max_lag,color='black')
            return g
        case (_,None):
            g = sns.FacetGrid(TRF,col=facet_col,row=facet_row,height=10)
            for (row_val), ax in g.axes_dict.items():
                max_lags = result_df[(result_df[facet_row]==row_val) ]['max_lag'].values
                tmp = TRF[(TRF[facet_row]==row_val)]
                sns.lineplot(data=tmp,x='lag',y='TRF',hue=color_variable,style=style_variable,ax=ax)
                for i,max_lag in enumerate(max_lags):
                    ax.axvline(max_lag,color='black')
            return g
        case (_,_):
            g = sns.FacetGrid(TRF,col=facet_col,row=facet_row,height=10)
            for (row_val, col_val), ax in g.axes_dict.items():
                max_lags = result_df[(result_df[facet_row]==row_val) & (result_df[facet_col]==col_val)]['max_lag'].values
                tmp = TRF[(TRF[facet_row]==row_val) & (TRF[facet_col]==col_val)]
                sns.lineplot(data=tmp,x='lag',y='TRF',hue=color_variable,style=style_variable,ax=ax)
                for i,max_lag in enumerate(max_lags):
                    ax.axvline(max_lag,color='black')
            return g



def plot_rasters_with_unified_scale(matrixs,plot_shape,save_path,ylabels=None,xlabels=None,titles=None):
    #matrixs: list of matrixs. Each matrix is a y by x matrix
    #plot_shape: (m,n) , which indicates that m by n subplots will be created.
    #ylabels: list of ylabels. The length of ylabels should be equal to m
    #xlabels: list of xlabels. The length of xlabels should be equal to n
    #save_path: path to save the figure

    #unify the colorbar
    vmax = -np.inf
    vmin = np.inf
    for matrix in matrixs:
        vmax = np.max([vmax,np.max(matrix)])
        vmin = np.min([vmin,np.min(matrix)])

    #plot each matrix
    fig,ax = plt.subplots(plot_shape[0],plot_shape[1],figsize=(plot_shape[1]*5,plot_shape[0]*5))

    for index,model_matrix in enumerate(matrixs):
        xindex=index%plot_shape[1]
        yindex=index//plot_shape[1]
        # assert yindex==index%plot_shape[0]
        # heatmap plot dnn_matrix
        sns.heatmap(model_matrix,ax=ax.flatten()[index],cmap='RdBu_r',cbar=False,vmin=vmin,vmax=vmax)
    
        ax.flatten()[index].set_title(titles[yindex])
        ax.flatten()[index].set_xlabel(xlabels[xindex])
        ax.flatten()[index].set_ylabel(ylabels[yindex])
        # ax.flatten()[index].set_yticks([])
        ax.flatten()[index].set_xticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    #example
    import glob
    import pickle
    import os
    data_dir = '/scratch/snormanh_lab/shared/Sigurd/dnn-feats-speechAll/pickle/'
    data_files = glob.glob(data_dir+'*.pkl')
    data_files.sort()
    matrixs = []
    for data_file in data_files:
        print(data_file)
        with open(data_file,'rb') as f:
            matrix = pickle.load(f)['input_after_preproc']
            #reshape 1,1,h,w to h,w
            matrix = matrix.reshape(matrix.shape[2],matrix.shape[3])
        matrixs.append(matrix)
    
    plot_shape = (3,1)
    save_path = 'test.png'
    xlabels = ['cochleagram']
    ylabels = titles =[os.path.splitext(os.path.basename(data_file))[0] for data_file in data_files]
    plot_rasters_with_unified_scale(matrixs,plot_shape,save_path,ylabels,xlabels,titles)
