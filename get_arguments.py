import argparse

def get_arguments_train():
    parser = argparse.ArgumentParser()
    
    #set directory
    parser.add_argument("data_directory", default="flowers")
    
    #save directory
    parser.add_argument("--save_dir", default="save_directory")
    
    #set architecture
    parser.add_argument("--arch", default="vgg13")
    
    #hyperparameters
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--hidden_units", default=512)
    parser.add_argument("--epochs", default=5)
    
    #gpu
    parser.add_argument("--gpu", action="store_const", const=True, default=False)
    
    return parser.parse_args()



def get_arguments_predict():
    parser = argparse.ArgumentParser()
    
    #set directory
    parser.add_argument("path")
    parser.add_argument("checkpoint", default="checkpoint.pth")
    
    #topk
    parser.add_argument("--topk", default=3)
    
    #category names
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--hidden_units", default=512)
    
    #gpu
    parser.add_argument("--gpu", action="store_const", const=True, default=False)
    
    return parser.parse_args()
    