import matplotlib.pyplot as plt
import seaborn as sn
import sys,os, scipy, glob
import numpy as np
import pandas as pd
import shutil
from IPython.display import display,HTML # to show the animation in Jupyter
from celluloid import Camera # getting the camera
from sklearn.decomposition import PCA, NMF
from scipy import stats
import matplotlib.image as mpimg

root_path_results = '/media/sf_Shardededboi/Bachelorboi/Phase-splitter/Results'


def get_in_situ_data(path):
    folder = path.split('/')
    folder = folder[-1]
    if not os.path.exists(root_path_results+"/"+folder):
        os.makedirs(root_path_results+"/"+folder)
    else:
        print("Folder name already taken")
        print("")
        print("Please rename your data folder or delete the existing folder from Results")
        answer = (input("Do you want me to delete content in Results/"+folder+": "))
        if answer == "yes" or answer == "Yes" or answer == "y" or answer == "Y" or answer == "ja" or answer == "Ja":
            try:
                dir = os.path.join(root_path_results+"/"+folder)
                if not os.path.exists(dir):
                    os.mkdir(dir)
                else:
                    shutil.rmtree(root_path_results+"/"+folder)
                    os.mkdir(root_path_results+"/"+folder)
            except:
                print("Results/"+folder + " is cleaned")
        elif answer == "No" or answer == "no" or answer == "n" or answer == "N" or answer == "nej" or answer == "Nej" or answer == "Nein" or answer == "nein":
            print("Nothing happend - files will be saved in the folder with the same name")
        else:
            print("Invalid input")
            sys.exit()
    print("Loading data")
    files = sorted(os.listdir(path))
    files = [file for file in files if file[0] != '.' or file[0] != '_']
    if '.gr' != None:
        files = [file for file in files if file[len('.gr')] != '.gr']
    y = []
    for i, file in enumerate(files):
        for j in range(100):
            try:
                ph = np.loadtxt(path + '/' + file, skiprows=j)
                break
            except:
                pass
        ph = ph.T
        if i == 0:
            x = ph[0]
            y.append(ph[1])
        else:
            y.append(ph[1])
    print("Your data is loaded correctly!")
    return np.array(x), np.array(y), folder

def Normalize(y):
    Y = (y - np.min(y)) / (np.max(y) - np.min(y))
    Y = np.array(Y).T
    return Y

def gen_corr(y,folder):
    fig, ax = plt.subplots(nrows=1, figsize=(12, 9))
    print('Creating correlation matrix')

    index = np.arange(len(y))
    df = pd.DataFrame(y.T, columns=index)

    corr = df.corr()
    mask = np.tril(np.ones_like(corr, dtype=bool))

    sn_ax = sn.heatmap(corr, mask=mask, cmap='magma', square=True,
                       cbar_kws={"pad": .15, 'label': 'Pearson correlation coefficient ', 'orientation': 'vertical'})

    ax = plt.gca()

    ax.tick_params(axis='y', labelrotation=0)
    plt.xlabel('Time [a.u]')
    ax.yaxis.set_label_position("right")
    plt.ylabel('Time [a.u]')
    plt.gca().invert_yaxis()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.set_xlabel('Time [a.u]')
    ax.set_ylabel('Time [a.u]')
    plt.tight_layout()
    plt.savefig(root_path_results+"/"+folder + "/" + 'Corr_mat.png', format='png', dpi=300)
    return None

def PCA_plot(Y,folder,n=None):
    print("Performing PCA")
    if n != None:
        n_components = n
    else:
        n_components = 5
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(Y)
    variance_exp_cumsum = pca.explained_variance_ratio_.round(2).cumsum()

    plt.style.use(bg_mpl_style)
    fig, ax = plt.subplots(nrows=1, figsize=(12,9))
    plt.plot(range(1, (n_components + 1)), variance_exp_cumsum, "o", color="firebrick")
    ax.set_xlabel('number of components')
    ax.set_ylabel('Variance Explained (\%)')  # ^{-1}
    plt.xlim(0, (n_components + 1))
    plt.ylim((variance_exp_cumsum[0] - 0.01), (variance_exp_cumsum[-1] + 0.01))
    plt.savefig(root_path_results + "/"+folder + "/" + 'PCA.png', format='png', dpi=300)
    return None
def plot(folder):
    img1 = mpimg.imread(root_path_results + "/"+folder + "/"  + 'Corr_mat.png')
    img2 = mpimg.imread(root_path_results + "/"+folder + "/"  + 'PCA.png')

    f = plt.figure(figsize=(30, 12))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.imshow(img1)
    ax.axis('off')
    ax2.imshow(img2)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
def phase_estimation(y):
    p_com = []
    pca = PCA(n_components=10)
    df_pca = pca.fit_transform(Normalize(y))
    variance_exp_cumsum = pca.explained_variance_ratio_.round(2).cumsum()
    varli = len(variance_exp_cumsum)
    for i, elem in enumerate(variance_exp_cumsum):
        nextelem = variance_exp_cumsum[(i + 1) % varli]
        if nextelem - elem > (variance_exp_cumsum[1] / 50):
            p_com.append(1)
    components = np.sum(p_com) + 1
    print("Phase-it's qualified guesses of phases in the data are: "+str(components))

def NMF_cal(x,Y,folder):
    nmf_com = int(input("Numbers of NMF components: "))
    print("Calculating NMF components")

    nmf = NMF(n_components=nmf_com, init="random", random_state=0, max_iter=20000)
    W = nmf.fit_transform(Y)
    plt.style.use(bg_mpl_style)
    fig, ax = plt.subplots(nrows=1, figsize=(12, 9))
    for i in range(np.shape(W)[1]):
        W[:, i] = (W[:, i] - np.min(W[:, i])) / (np.max(W[:, i]) - np.min(W[:, i]))
    for j in range(nmf_com):
        plt.plot(x, W.T[j] - j, label="NMF Component Number " + str(int(j + 1)))
    """
    if nmf_com == 2:
        plt.plot(x, Y.T[0] - 1, label="First Experimental Frame")
        plt.plot(x, Y.T[-1] - 2, label="Last Experimental Frame")
    else:
        plt.plot(x, Y.T[0] - 1, label="First Experimental Frame")
        plt.plot(x, Y.T[int(len(y) / 2)] - 2, label="Middle Experimental Frame")
        plt.plot(x, Y.T[-1] - 3, label="Last Experimental Frame")
    """
    ax.set_xlabel('r[Å]')
    ax.set_ylabel('G(r) [a.u]')
    ax.set_yticklabels('')
    plt.legend(loc="upper right")
    plt.xlim(x[0],x[-1])
    plt.savefig(root_path_results + "/"+folder + "/"  + 'NMF.png', format='png', dpi=300)

    for i in range(np.shape(W)[1]):
        z = W.T[i]
        np.savetxt(root_path_results + "/"+folder + "/"  + 'NMF_com_' + str(i + 1) + '.gr', np.column_stack([x, z]))
    return None
def insitu_plot(x, y,folder):
    print("Making contour plot")
    plt.style.use(bg_mpl_style)

    frame = np.arange(len(y))
    fig, ax = plt.subplots(nrows=1, figsize=(12,9))

    cmap = plt.get_cmap('magma')
    X, Y = np.meshgrid(x, frame)
    Z = (y)
    thisPlot = plt.pcolormesh(X, Y, Z, cmap=cmap)

    plt.xlabel('r[Å]') #
    plt.ylabel('Time [a.u]')

    cbar = plt.colorbar(thisPlot, ticks=[0], aspect=10)
    mx = np.max(y)
    mm = np.min(y)
    cbar.set_ticks([mm, mx])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('G(r)', rotation=270)
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')

    cbar_ax = fig.axes[-1]

    ax.set_xlabel('r[Å]')
    ax.set_ylabel('Time [a.u]')  # ^{-1}

    plt.tight_layout()
    plt.savefig(root_path_results + "/"+folder + "/"  + 'insitu_plot.png', format='png', dpi=300)
    return None

def Dynamicgif(x,y,folder):
    print("Making gif")
    plt.style.use(bg_mpl_style)
    fig, ax = plt.subplots(nrows=1, figsize=(12,9))
    camera = Camera(fig)
    for i in range(len(y)):
        x_t = x
        y_t = y[i]
        ax.plot(x_t, y_t, c='red')
        ax.text(0.5, 1.01, "Frame = " + str(i + 1) + "/" + str(len(y)), transform=ax.transAxes)
        camera.snap()
    ax.set_xlabel('r[Å]')
    ax.set_ylabel('G(r) [a.u.]')  # ^{-1}
    animation = camera.animate()
    display(HTML(animation.to_html5_video()))
    animation.save(root_path_results + "/"+folder + "/"  + 'dynamic_transition.gif', writer='ffmpeg', fps=60, dpi=100, metadata={'title': 'test'});
    return None

def pearson_nmf(x,y,folder):
    NMF = []
    pear_corr = []

    files = sorted(glob.glob(root_path_results + "/"+folder + "/"  + "/*.gr"))

    for pdf in files:
        PDF = np.loadtxt(pdf)
        gr = PDF[:, 1]
        NMF.append(gr)
    Z = np.array(NMF)

    for i in range(len(Z)):
        for j in range(len((y))):
            a = scipy.stats.pearsonr(Normalize(y[j]), Z[i])[0]
            pear_corr.append(a)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    p_corr = list(chunks(pear_corr, len((y))))

    plt.style.use(bg_mpl_style)
    fig, ax = plt.subplots(nrows=1, figsize=(12, 9))
    for t in range(len(Z)):
        plt.plot(x, Z[t] - 1.5 * t, 'o', lw=3,
                 label="NMF com " + str(int(t + 1)) + " - " + str(np.round(np.max(p_corr[t]), 2)))
        plt.plot(x, Normalize(y[np.argmax(p_corr[t])]) - 1.5 * t, 'k', lw=2)
        plt.plot(x, (Z[t] - Normalize(y[np.argmax(p_corr[t])])) - 1.5 * t - 0.25, 'k')
        ax.hlines(np.mean((Z[t] - Normalize(y[np.argmax(p_corr[t])]))) - 1.5 * t - 0.25, x[0], x[-1], colors="k",
                  linestyles="dotted")
    plt.plot([], [], 'k', label="Difference")
    ax.set_xlabel('r[Å]')
    ax.set_ylabel('G(r) [a.u]')
    ax.set_yticklabels('')
    plt.legend(loc="upper right")
    plt.xlim(x[0], x[-1])
    plt.tight_layout()
    plt.savefig(root_path_results + "/"+folder + "/" + 'nmf_pdf_relation.png', format='png', dpi=300)
    return None
def contour_nmf(x,y,folder):
    NMF = []
    pear_corr = []

    files = sorted(glob.glob(root_path_results + "/"+folder + "/"  + "/*.gr"))

    for pdf in files:
        PDF = np.loadtxt(pdf)
        gr = PDF[:, 1]
        NMF.append(gr)
    Z = np.array(NMF)

    for i in range(len(Z)):
        for j in range(len(y)):
            a = scipy.stats.pearsonr(Normalize(y[j]), Z[i])[0]
            pear_corr.append(a)

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    p_corr = list(chunks(pear_corr, len(y)))
    line_list = []
    for t in range(len(Z)):
        line_list.append(np.argmax(p_corr[t]))
    #line_list.sort()

    plt.style.use(bg_mpl_style)

    frame = np.arange(len(y))
    fig, ax = plt.subplots(nrows=1, figsize=(12, 9))

    cmap = plt.get_cmap('magma')
    X, Y = np.meshgrid(x, frame)
    Z = (y)
    thisPlot = plt.pcolormesh(X, Y, Z, cmap=cmap)

    plt.xlabel('r[Å]')  #
    plt.ylabel('Time [min]')

    cbar = plt.colorbar(thisPlot, ticks=[0], aspect=10)
    mx = np.max(y)
    mm = np.min(y)
    cbar.set_ticks([mm, mx])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('G(r)', rotation=270)
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')

    cbar_ax = fig.axes[-1]

    ax.set_xlabel('r[Å]')
    ax.set_ylabel('Time [a.u]')  # ^{-1}
    color_cycle = ["b", "r", "g", "c", "m", "y", "k"]
    for i in range(len(line_list)):
        ax.hlines(line_list[i] - len(y) / 100, colors=color_cycle[i], lw=3, *ax.get_xlim())

    plt.tight_layout()
    plt.savefig(root_path_results + "/" +folder + "/" + 'contour_nmf_relation.png', format='png', dpi=300)
    return None

def plot_2(folder):
    img1 = mpimg.imread(root_path_results + "/"+folder + "/" + 'contour_nmf_relation.png')
    img2 = mpimg.imread(root_path_results + "/"+folder + "/" + 'nmf_pdf_relation.png')

    f = plt.figure(figsize=(30, 12))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.imshow(img1)
    ax.axis('off')
    ax2.imshow(img2)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
