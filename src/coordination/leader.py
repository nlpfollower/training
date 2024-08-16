import socket
import json
import time

def execute_instruction(instruction):
    # This function would contain the logic to execute the received instruction
    # For now, we'll just print the instruction and return a dummy result
    print(f"Executing instruction: {instruction}")
    return {"result": f"Executed {instruction['action']}"}


def main():
    host = '127.0.0.1'  # localhost
    port = 65432  # arbitrary non-privileged port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()

        print(f"Python server listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                data = conn.recv(1024)
                if not data:
                    continue

                instruction = json.loads(data.decode('utf-8'))
                result = execute_instruction(instruction)

                conn.sendall(json.dumps(result).encode('utf-8'))


if __name__ == "__main__":
    main()