from flask import Flask, jsonify, request, render_template,make_response,send_file
import torch
import torch.nn.functional as F
import pandas as pd
from flask_cors import CORS
from fileinput import filename

app = Flask(__name__)
CORS(app)


def precoss(a):
    a[0] = a[0] - 20
    a[2] = a[2] + 20
    a[3] = a[3] - 20
    a[5] = a[5] + 20
    a[6] = a[6] - 20
    a[8] = a[8] + 20

    a[9] = a[9] + 20
    a[10] = a[10] + 20
    a[11] = a[11] + 20
    a[15] = a[15] - 20
    a[16] = a[16] - 20
    a[17] = a[17] - 20
    b = [a[0], a[9], a[1], a[10], a[2], a[11], a[3], a[12], a[4], a[13], a[5], a[14], a[6], a[15], a[7], a[16], a[8],
         a[17]]
    return b
    # ... existing code for data preprocessing ...


class Net(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Net, self).__init__()
        # 将输入的4维映射到100维高维空间
        self.hidder1 = torch.nn.Linear(in_feature, 512)
        self.hidder2 = torch.nn.Linear(512, 256)
        # 将映射的100维参数空间映射回输出的3维空间
        self.out = torch.nn.Linear(256, out_feature)

        

    def forward(self, x):
        x = F.relu(self.hidder1(x))
        x = F.relu(self.hidder2(x))
        x = self.out(x)
        return x


net = torch.load("model.pth").cpu()
net.eval()


@app.route('/')
def index():
    return "<h1>Hello,flask!</h1>"


def recive_data(data):
    a1 = int(data.get('top_left1'))
    a2 = int(data.get('top1'))
    a3 = int(data.get('top_right1'))
    a4 = int(data.get('left1'))
    a5 = int(data.get('center1'))
    a6 = int(data.get('right1'))
    a7 = int(data.get('bottom_left1'))
    a8 = int(data.get('bottom1'))
    a9 = int(data.get('bottom_right1'))

    b1 = int(data.get('top_left2'))
    b2 = int(data.get('top2'))
    b3 = int(data.get('top_right2'))
    b4 = int(data.get('left2'))
    b5 = int(data.get('center2'))
    b6 = int(data.get('right2'))
    b7 = int(data.get('bottom_left2'))
    b8 = int(data.get('bottom2'))
    b9 = int(data.get('bottom_right2'))
    return [a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8, b9]


def zhenduan(out):
    str=''
    b_out = '{:08b}'.format(out)
    if out == 0:
        str += "该眼各条肌肉力量正常\n"
    if b_out[2] == '1':
        str += "上直肌力弱\n"
    if b_out[3] == '1':
        str += "下直肌力弱\n"
    if b_out[4] == '1':
        str += "内直肌力弱\n"
    if b_out[5] == '1':
        str += "下斜肌力弱\n"
    if b_out[6] == '1':
        str += "上斜肌力弱\n"
    if b_out[7] == '1':
        str += "外直肌力弱\n"

    return str



@app.route('/file', methods=['POST'])
def file():
    file = request.files['file']
    # save file in local directory
    file.save("uplod/" + file.filename)

    # Parse the data as a Pandas DataFrame type
    data = pd.read_excel("uplod/" + file.filename, header=[0, 1, 2])
    data.insert(data.shape[1], '真实类别', '')
    data.insert(data.shape[1], '预测类别', '')
    data.insert(data.shape[1],'诊断结果','')
    print(data.shape)
    for i in range(data.shape[0]):
        b = precoss(data.iloc[i, :18])
        b_in = torch.FloatTensor(b)
        out = torch.argmax(net(b_in))
        number = data.iloc[i, 23] * 1 + data.iloc[i, 22] * 2 + data.iloc[i, 21] * 4 + data.iloc[i, 20] * 8 + data.iloc[
            i, 19] * 16 + data.iloc[i, 18] * 32
        zd=zhenduan(out)
        data.iloc[i, 24] = str(number)
        data.iloc[i, 25] = str(int(out.data))
        data.iloc[i,26]=zd
    data.to_excel('分析结果.xlsx')
    response = make_response(send_file('分析结果.xlsx'))
    response.headers["Content-Disposition"] = "attachment; filename={}".format("分析结果.xlsx".encode().decode('latin-1'))
    return response
    # return data.to_html()
    # Return HTML snippet that will render the table


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        print(data)
        b = precoss(recive_data(data))
        b_in = torch.FloatTensor(b)
        out = torch.argmax(net(b_in))
        b_out = '{:08b}'.format(out)
        str = '<h2>诊断结果为:</h2>\n'
        if out == 0:
            str += "<h2>该眼各条肌肉力量正常</h2>\n"
        if b_out[2] == '1':
            str += "<h2>上直肌力弱</h2>\n"
        if b_out[3] == '1':
            str += "<h2>下直肌力弱</h2>\n"
        if b_out[4] == '1':
            str += "<h2>内直肌力弱</h2>\n"
        if b_out[5] == '1':
            str += "<h2>下斜肌力弱</h2>\n"
        if b_out[6] == '1':
            str += "<h2>上斜肌力弱</h2>\n"
        if b_out[7] == '1':
            str += "<h2>外直肌力弱</h2>\n"

        return str
    else:
        return '提交数据需要post请求'


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=9999)
