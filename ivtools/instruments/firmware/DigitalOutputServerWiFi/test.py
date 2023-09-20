import socket
#host = "192.168.0.73"
host = '192.168.4.1'
port = 1337

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

# apparently won't connect again until s.close()

# careful, \n is 0b00001010, could be a nasty bug

def send_data(databinary, nbytes=3):
    databytes = databinary.to_bytes(nbytes, 'big') # big or little?
    s.send(databytes)
    s.send(b'\n')

for i in range(10):
    data = 0b0110111101101110
    send_data(data)
    data |= 1
    send_data(data)
