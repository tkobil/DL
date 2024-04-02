import pickle
import torch
from model import ResNet18

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preprocess_data():
    # Turn 10000x3072 into
    # 10000 x 3 x 32 x 32
    filename = "cifar_test_nolabels.pkl"
    unpickled = unpickle(filename)
    return unpickled
        



def run_inference(model, device):
    model.eval()
    data = preprocess_data()
    
    
    with open('output_model.csv', 'w') as outfile:
        outfile.write('ID,Labels\n')
    
        for i in range(10000):
            if (i % 100) == 0:
                print(i)
                
            input = data[b'data'][i].reshape(3, 32, 32) / 255 # normalize image
            input = torch.from_numpy(input.astype('float32')).unsqueeze(dim=0).to(device)
            output = model(input)
            _, prediction = output.max(1)
            id = data[b'ids'][i]
            outfile.write(f'{id},{prediction.item()}\n')
    
    

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    model = ResNet18()
    model.load_state_dict(torch.load('model_optimizer=SGD_lr=0.03_momentum=0.9_weightdecay=0.0001_numepochs=9_scheduler=ExponentialLR.pt'))
    model.to(device)
    
    run_inference(model, device)
            
    

if __name__ == "__main__":
    main()