from flask import Flask, render_template, request
from chatbot.chatbot import get_bot_resp
from flask import render_template

official='If the reponse got is insatisfactory you can contact us at 079 2327 5060'
space="\n"
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    ans1 = ""
    param='cosine'
    userText = request.args.get('msg')
    ans,flag=get_bot_resp(userText,param)
    if(ans!=None):
        #print('Ans1 returning from here',ans1)
        ans='{} {} {}'.format(ans,'\n',official)
        return  ans
    else:
        return official

def run_app():
    app.run(debug=True)