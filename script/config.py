from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms, datasets
from skimage.io import imread

########################################################################################################################
test_idx = 27

########################################################################################################################
#base_root = "/home/compu/ymh/contrastive/"
#base_root = "/DATA/ymh/contrastive/"
base_root = "C://유민형//개인 연구//Constrastive learning//"

celebA_root = "/DATA/ymh/contrastive/data/CelebA"# "/home/compu/ymh/contrastive/data/CelebA" 

g1_pre_savefile = base_root + "test" + str(test_idx) + "_g1.pth"
d1_pre_savefile = base_root + "test" + str(test_idx) + "_d1.pth"
g2_pre_savefile = base_root + "test" + str(test_idx) + "_g2.pth"
d2_pre_savefile = base_root + "test" + str(test_idx) + "_d2.pth"

generator_pre_savefile = base_root + "test" + str(test_idx) + "_generator.pth"
discriminator_pre_savefile = base_root + "test" + str(test_idx) + "_discriminator.pth"
encoder_pre_savefile = base_root + "test" + str(test_idx) + "_encoder.pth"
encoder1_pre_savefile = base_root + "test" + str(test_idx) + "_encoder1.pth"
encoder2_pre_savefile = base_root + "test" + str(test_idx) + "_encoder2.pth"
decoder_pre_savefile = base_root + "test" + str(test_idx) + "_decoder.pth"
pre_model_save = 10
pre_image_save = 1
pre_image_savefile = base_root + "sample6/epoch_"

########################################################################################################################
pre_test_idx = 26
generator_load = base_root + "test" + str(pre_test_idx) + "_generator.pth"
discriminator_load = base_root + "test" + str(pre_test_idx) + "_discriminator.pth"

g1_load = base_root + "test" + str(pre_test_idx) + "_g1.pth"
d1_load = base_root + "test" + str(pre_test_idx) + "_d1.pth"
g2_load = base_root + "test" + str(pre_test_idx) + "_g2.pth"
d2_load = base_root + "test" + str(pre_test_idx) + "_d2.pth"

########################################################################################################################
noise_var = 0.0001
lr = 0.0002
b1 = 0.5
b2 = 0.999

########################################################################################################################
pre_epoch = 1000
epoch = 1000

########################################################################################################################
num_data = 60000 # 202599 #

batch_size = 256
batch_num = num_data // batch_size

pre_batch_size = 512 #128*6
pre_batch_num = num_data // pre_batch_size

########################################################################################################################
img_shape = (1, 32, 32)
latent_dim = (100, 1, 1)
mean = 0.5
std = 0.5
########################################################################################################################

color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
transform_pretrain = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize(img_shape[1]),
         transforms.RandomCrop(img_shape[1]),
         transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
         transforms.RandomApply([color_jitter], p=0.8),
         transforms.ToTensor(),
         transforms.Normalize((mean,), (std,)) #transforms.Normalize((mean,mean,mean), (std,std,std)) # 
     ])

transform_train = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize(img_shape[1]),
         transforms.ToTensor(),
         transforms.Normalize((mean,), (std,)) #transforms.Normalize((mean,mean,mean), (std,std,std)) # 
     ])


########################################################################################################################
class Augmented_MNIST(Dataset):
    def __init__(self, root, aug_transform):
        super().__init__()

        self.aug_transform = aug_transform
        
        save_file = os.path.join(root, 'Augmented_MNIST.npy')
        
        if os.path.isfile(save_file):
            self.mnist = np.load(save_file)
            print("File is loaded.")
        
        else:
            train_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor()])
                        ),
                batch_size=1, shuffle=False)
            
            self.mnist = np.empty((60000,28,28,1))
            
            for idx, (x, _) in enumerate(train_loader):
                x = x*255
                x = x.numpy().reshape(28,28,1)
                self.mnist[idx] = x
                print("[%d/60000]" % (idx+1))
                
            save_file = os.path.join(root, 'Augmented_MNIST')
            np.save(save_file, self.mnist)
                
        self.len = 60000
        self.mnist = self.mnist.astype('uint8')
        
    def __getitem__(self, index):
        
        img = self.mnist[index]
                   
        aug1 = self.aug_transform(img)
        aug2 = self.aug_transform(img)
        
        return aug1, aug2
    
    def __len__(self):
        return self.len

########################################################################################################################
class CelebA_Image(Dataset):
    def __init__(self, root, image_path=None, transform=None):
        super().__init__()
        
        self.transform=transform
        self.len = 202599
        
        save_file1 = os.path.join(root, 'CelebA_Image_part1.npy')
        save_file2 = os.path.join(root, 'CelebA_Image_part2.npy')
        save_file3 = os.path.join(root, 'CelebA_Image_part3.npy')
        save_file4 = os.path.join(root, 'CelebA_Image_part4.npy')
        save_file5 = os.path.join(root, 'CelebA_Image_part5.npy')
        
        if os.path.isfile(save_file1):
            self.CelebA_part1 = np.load(save_file1)
            print("Part 1 is loaded.")
        else:
            self.CelebA_part1 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx+1)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part1[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file1 = os.path.join(root, 'CelebA_Image_part1.npy')
            np.save(save_file1, self.CelebA_part1)
            print("Part1 is saved.")
        
        if os.path.isfile(save_file2):
            self.CelebA_part2 = np.load(save_file2)
            print("Part 2 is loaded.")
        else:
            self.CelebA_part2 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 40000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part2[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file2 = os.path.join(root, 'CelebA_Image_part2.npy')
            np.save(save_file2, self.CelebA_part2)
            print("Part2 is saved.")
        
        if os.path.isfile(save_file3):
            self.CelebA_part3 = np.load(save_file3)
            print("Part 3 is loaded.")
        else:
            self.CelebA_part3 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 80000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part3[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file3 = os.path.join(root, 'CelebA_Image_part3.npy')
            np.save(save_file3, self.CelebA_part3)
            print("Part3 is saved.")
        
        if os.path.isfile(save_file4):
            self.CelebA_part4 = np.load(save_file4)
            print("Part 4 is loaded.")
        else:
            self.CelebA_part4 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 120000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part4[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file4 = os.path.join(root, 'CelebA_Image_part4.npy')
            np.save(save_file4, self.CelebA_part4)
            print("Part4 is saved.")
        
        if os.path.isfile(save_file5):
            self.CelebA_part5 = np.load(save_file5)
            print("Part 5 is loaded.")
        else:
            self.CelebA_part5 = np.empty((42599,218,178,3), dtype='uint8')
            
            for idx in range(42599):
                image_name = "%06d.jpg" % (idx + 1 + 160000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part5[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 42599))
                
            save_file5 = os.path.join(root, 'CelebA_Image_part5.npy')
            np.save(save_file5, self.CelebA_part5)
            print("Part5 is saved.")
        
        self.CelebA = np.concatenate((self.CelebA_part1, self.CelebA_part2, self.CelebA_part3, self.CelebA_part4, self.CelebA_part5), axis=0)
        print("Dataset is ready.")
        
        #self.CelebA = self.CelebA.astype('uint8')
        
    def __getitem__(self, index):
        
        img = self.CelebA[index]
        
        if self.transform is not None:
            img = self.transform(img)
        else:  
            img = img.transpose(2,0,1)
        
        return img
    
    def __len__(self):
        return self.len

########################################################################################################################
class Augmented_CelebA(Dataset):
    def __init__(self, root, transform, image_path=None):
        super().__init__()
        
        self.transform=transform
        self.len = 202599
        
        save_file1 = os.path.join(root, 'CelebA_Image_part1.npy')
        save_file2 = os.path.join(root, 'CelebA_Image_part2.npy')
        save_file3 = os.path.join(root, 'CelebA_Image_part3.npy')
        save_file4 = os.path.join(root, 'CelebA_Image_part4.npy')
        save_file5 = os.path.join(root, 'CelebA_Image_part5.npy')
        
        if os.path.isfile(save_file1):
            self.CelebA_part1 = np.load(save_file1)
            print("Part 1 is loaded.")
        else:
            self.CelebA_part1 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx+1)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part1[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file1 = os.path.join(root, 'CelebA_Image_part1.npy')
            np.save(save_file1, self.CelebA_part1)
            print("Part1 is saved.")
        
        if os.path.isfile(save_file2):
            self.CelebA_part2 = np.load(save_file2)
            print("Part 2 is loaded.")
        else:
            self.CelebA_part2 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 40000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part2[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file2 = os.path.join(root, 'CelebA_Image_part2.npy')
            np.save(save_file2, self.CelebA_part2)
            print("Part2 is saved.")
        
        if os.path.isfile(save_file3):
            self.CelebA_part3 = np.load(save_file3)
            print("Part 3 is loaded.")
        else:
            self.CelebA_part3 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 80000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part3[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file3 = os.path.join(root, 'CelebA_Image_part3.npy')
            np.save(save_file3, self.CelebA_part3)
            print("Part3 is saved.")
        
        if os.path.isfile(save_file4):
            self.CelebA_part4 = np.load(save_file4)
            print("Part 4 is loaded.")
        else:
            self.CelebA_part4 = np.empty((40000,218,178,3), dtype='uint8')
            
            for idx in range(40000):
                image_name = "%06d.jpg" % (idx + 1 + 120000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part4[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 40000))
                
            save_file4 = os.path.join(root, 'CelebA_Image_part4.npy')
            np.save(save_file4, self.CelebA_part4)
            print("Part4 is saved.")
        
        if os.path.isfile(save_file5):
            self.CelebA_part5 = np.load(save_file5)
            print("Part 5 is loaded.")
        else:
            self.CelebA_part5 = np.empty((42599,218,178,3), dtype='uint8')
            
            for idx in range(42599):
                image_name = "%06d.jpg" % (idx + 1 + 160000)
                img = imread(os.path.join(image_path, image_name))
                img = np.array([img])
                self.CelebA_part5[idx] = img
                if (idx+1) % 100 == 0:
                    print('Done: {0}/{1} images'.format((idx+1), 42599))
                
            save_file5 = os.path.join(root, 'CelebA_Image_part5.npy')
            np.save(save_file5, self.CelebA_part5)
            print("Part5 is saved.")
        
        self.CelebA = np.concatenate((self.CelebA_part1, self.CelebA_part2, self.CelebA_part3, self.CelebA_part4, self.CelebA_part5), axis=0)
        print("Dataset is ready.")
        
        #self.CelebA = self.CelebA.astype('uint8')
        
    def __getitem__(self, index):
        
        img = self.CelebA[index]
        
        aug1 = self.transform(img)
        aug2 = self.transform(img)
        
        return aug1, aug2
    
    def __len__(self):
        return self.len




