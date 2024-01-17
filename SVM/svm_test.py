from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from pylab import mpl
print(mpl.get_cachedir())
mpl.rcParams["font.sans-serif"]=["SimHei"]

def del_data(file_path):
    data=pd.read_csv(file_path)
    print(data)
    data.iloc[:,0]=data.iloc[:,0]-20
    data.iloc[:,2] = data.iloc[:,2] + 20
    data.iloc[:,3] = data.iloc[:,3] - 20
    data.iloc[:,5] = data.iloc[:,5] + 20
    data.iloc[:,6] = data.iloc[:,6] - 20
    data.iloc[:,8] = data.iloc[:,8] + 20

    data.iloc[:,9] = data.iloc[:,9] + 20
    data.iloc[:,10] = data.iloc[:,10] + 20
    data.iloc[:,11] = data.iloc[:,11] + 20
    data.iloc[:,15] = data.iloc[:,15] - 20
    data.iloc[:,16] = data.iloc[:,16] - 20
    data.iloc[:,17] = data.iloc[:,17] - 20
    data.iloc[:,24] = data.iloc[:,18] *32 +data.iloc[:,19] *16+data.iloc[:,20] *8+data.iloc[:,21] *4+data.iloc[:,22] *2+data.iloc[:,23]
    X=data.iloc[:,:18]
    Y=data.iloc[:,24]
    return X,Y


if __name__ == "__main__":
    train_data_path = "/work/xiaolu/uplod/data1127.csv"
    test_data_path = "/work/xiaolu/uplod/data226.csv"
    x_train,y_train=del_data(train_data_path)
    x_test,y_test=del_data(test_data_path)
    model=svm.SVC(decision_function_shape='ovr')
    print("开始训练")
    model.fit(x_train,y_train)
    print("训练完成")
    pred_test=model.predict(x_test)
    acu=accuracy_score(y_test,pred_test)
    print("acu:",acu)
    #0.814
    c = confusion_matrix(y_test, pred_test)
    print(c)
    # cm_display=ConfusionMatrixDisplay(c,display_labels=label).plot()
    fig, ax = plot_confusion_matrix(conf_mat=c, show_absolute=False, show_normed=True, colorbar=True, figsize=(24, 18),
                                    )
    plt.savefig("cm_svm.png")
