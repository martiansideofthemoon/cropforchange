from socket import *

def createServer():
    serversocket = socket(AF_INET, SOCK_STREAM)
    serversocket.bind(('localhost',9000))
    serversocket.listen(5)
    while(1):
        (clientsocket, address) = serversocket.accept()
        print address
        clientsocket.send("Content-type:text/html\n\n<html>Hello World</html>")

createServer()
