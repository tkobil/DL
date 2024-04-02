import pickle
import torch

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
        

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    model = torch.load('model_optimizer=SGD_lr=0.03_momentum=0.9_weightdecay=0.0005_numepochs=8_scheduler=None')
    import pdb;pdb.set_trace()
    model.eval()
    data = preprocess_data()
    
    
    with open('output_model.csv', 'w') as outfile:
        outfile.write('ID,Labels\n')
    
        for i in range(10000):
            if (i % 100) == 0:
                print(i)
            input = torch.from_numpy(data[b'data'][i].reshape(3, 32, 32)).float().unsqueeze(dim=0).to(device)
            output = model(input)
            _, prediction = output.max(1)
            id = data[b'ids'][i]
            outfile.write(f'{id},{prediction.item()}\n')
    

if __name__ == "__main__":
    main()