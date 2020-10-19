import socket
import threading
import model,exercise,workout

DEBUG = 1

def ReceiveInstruction(socket):

    '''
    NOTE: We need to establish how a complete command looks like. 
    '''
    try:

        instruction = socket.recv(4096)
        #do the function(instruction)
        



        

    except:
        #Handle exception
        exit()

    #Parse instruction



    # cmd = json.dumps(instruction)
    

def InitializeModel():
    
    #TODO
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224


    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


def main():

    #initialize model
    
    camera = Camera()
    
    HOST = '127.0.0.1'
    PORT = 42069
    pool = 3

    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.bind((HOST,PORT)) #HOST is possibly ''
    s.listen(pool)

    while True:
        
        conn, addr = s.accept()
        if DEBUG:
            print(f"{addr} connected!\n")
        t = threading.Thread(ReceiveInstruction,conn)
        t.start()


        
       

 

        
    
        




    return 0

if __name__ == '__main__':
    sys.exit(main())