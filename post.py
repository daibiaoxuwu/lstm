#!coding:utf-8
import httplib
import urllib

reqheaders={'Content-type':'application/x-www-form-urlencoded',
'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
'Host':'www.renren.com',
'Origin':'http://zhichang.renren.com',
'Referer':'http://zhichang.renren.com',
'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1',}

#定义post的参数

reqdata={'email':'xxxx@xxx.com',
'password':'xxxx',
'autoLogin':'on',
'origURL':'http://zhichang.renren.com/?login_state=rr',
'domain':'renren.com'
}

#对请求参数进行编码

data=urllib.urlencode(reqdata)

#利用httplib库模拟接口请求

#先连接到人人

conn=httplib.HTTPConnection('renren.com')

#提交登录的post请求
conn.request('POST', '/PLogin.do', data, reqheaders)

#获取服务器的返回
res=conn.getresponse()

#打印服务器返回的状态
print(res.status)

#以dictionary形式答应服务器返回的 response header

print(res.msg)
#打印服务器返回请求头中设置的cookie
print(res.getheader('Set-Cookie'))
