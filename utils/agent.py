from utils.models import *
from utils.tools import *
import os
import imageio
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

from tqdm.notebook import tqdm
from config import *

import glob
from PIL import Image

class Agent():
    def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False, model_name='vgg16', n_actions=9):
        # basic settings
        self.n_actions = n_actions                       # total number of actions
        screen_height, screen_width = 224, 224   # size of resized images
        self.GAMMA = 0.900                       # decay weight
        self.EPS = 1                             # initial epsilon value, decayed every epoch
        self.alpha = alpha                       # €[0, 1]  Scaling factor
        self.nu = nu                             # Reward of Trigger
        self.threshold = threshold               # threshold of IoU to consider as True detection
        self.actions_history = None              # action history vector as record, later initialized in train/predict
        self.steps_done = 0                      # to count how many steps it used to compute the final bdbox
        
        # networks
        self.classe = classe                     # which class this agent is working on
        self.save_path = SAVE_MODEL_PATH         # path to save network
        self.model_name = model_name             # which model to use for feature extractor 'vgg16' or 'resnet50' or ...
        self.feature_extractor = FeatureExtractor(network=self.model_name)
        self.feature_extractor.eval()            # a pre-trained CNN model as feature extractor
        
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions, history_length=9)
        else:
            self.policy_net = self.load_network() # policy net - DQN, inputs state vector, outputs q value for each action
        
        self.target_net = DQN(screen_height, screen_width, self.n_actions,  history_length=9)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()                    # target net - same DQN as policy net, works as frozen net to compute loss
                                                  # initialize as the same as policy net, use eval to disable Dropout
            
        # training settings
        self.BATCH_SIZE = 128                    # batch size
        self.num_episodes = num_episodes         # number of total episodes
        self.memory = ReplayMemory(10000)        # experience memory object
        self.TARGET_UPDATE = 1                   # frequence of update target net
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)  # optimizer
        
        if use_cuda:
            self.feature_extractor = self.feature_extractor.cuda()
            self.target_net = self.target_net.cuda()
            self.policy_net = self.policy_net.cuda()
        
        ## newly added
        self.current_coord = [0,224,0,224]

    def save_network(self):
        torch.save(self.policy_net, self.save_path + "_" + self.model_name + "_" +self.classe)
        print('Saved')

    def load_network(self):
        if not use_cuda:
            return torch.load(self.save_path + "_" + self.model_name + "_" + self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path + "_" + self.model_name + "_" + self.classe)
    
    #############################
    # 1. Functions to compute reward
    def intersection_over_union(self, box1, box2):
        """
        Compute IoU value over two bounding boxes
        Each box is represented by four elements vector: (left, right, top, bottom)
        Origin point of image system is on the top left
        """
        box1_left, box1_right, box1_top, box1_bottom = box1
        box2_left, box2_right, box2_top, box2_bottom = box2
        
        inter_top = max(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = min(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_bottom - inter_top)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

    def compute_reward(self, actual_state, previous_state, ground_truth):
        """
        Compute the reward based on IoU before and after an action (not trigger)
        The reward will be +1 if IoU increases, and -1 if decreases or stops
        ----------
        Argument:
        actual_state   - new bounding box after action
        previous_state - old boudning box
        ground_truth   - ground truth bounding box of current object
        *all bounding boxes comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        reward         - +1/-1 depends on difference between IoUs
        """
        res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1
    
    def compute_trigger_reward(self, actual_state, ground_truth):
        """
        Compute the reward based on final IoU before *trigger*
        The reward will be +nu if final IoU is larger than threshold, and -nu if not
        ----------
        Argument:
        actual_state - final bounding box before trigger
        ground_truth - ground truth bounding box of current object
        *all bounding boxes comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        reward       - +nu/-nu depends on final IoU
        """
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>=self.threshold:
            return self.nu
        return -1*self.nu
      
        
    
    ###########################
    # 2. Functions to get actions 
    def calculate_position_box(self, current_coord, action):
        """
        Calculate new coordinate based on current coordinate and taken action.
        ----------
        Argument:
        current_coord - the current coordinate of this agent, should comes in four elements vector (left, right, top, bottom)
        action        - the index of taken action, should between 0-8
        ----------
        Return:
        new_coord     - the coordinate after taking the action, also four elements vector
        """
        real_x_min, real_x_max, real_y_min, real_y_max = current_coord

        alpha_h = self.alpha * (real_y_max - real_y_min)
        alpha_w = self.alpha * (real_x_max - real_x_min)
        if action == 1: # Right
            real_x_min += alpha_w
            real_x_max += alpha_w
        if action == 2: # Left
            real_x_min -= alpha_w
            real_x_max -= alpha_w
        if action == 3: # Up 
            real_y_min -= alpha_h
            real_y_max -= alpha_h
        if action == 4: # Down
            real_y_min += alpha_h
            real_y_max += alpha_h
        if action == 5: # Bigger
            real_y_min -= alpha_h
            real_y_max += alpha_h
            real_x_min -= alpha_w
            real_x_max += alpha_w
        if action == 6: # Smaller
            real_y_min += alpha_h
            real_y_max -= alpha_h
            real_x_min += alpha_w
            real_x_max -= alpha_w
        if action == 7: # Fatter
            real_y_min += alpha_h
            real_y_max -= alpha_h
        if action == 8: # Taller
            real_x_min += alpha_w
            real_x_max -= alpha_w
                
        real_x_min = self.rewrap(real_x_min)
        real_x_max = self.rewrap(real_x_max)
        real_y_min = self.rewrap(real_y_min)
        real_y_max = self.rewrap(real_y_max)
        
        return [real_x_min, real_x_max, real_y_min, real_y_max]
    
    def get_best_next_action(self, current_coord, ground_truth):
        """
        Given actions, traverse every possible action, cluster them into positive actions and negative actions
        Then randomly choose one positive actions if exist, or choose one negtive actions anyways
        It is used for epsilon-greedy policy
        ----------
        Argument:
        current_coord - the current coordinate of this agent, should comes in four elements vector (left, right, top, bottom)
        ----------
        Return:
        An action index that represents the best action next
        """
        positive_actions = []
        negative_actions = []
        for i in range(0, self.n_actions):
            new_equivalent_coord = self.calculate_position_box(current_coord, i)
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, current_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord, ground_truth)
            
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)

    def select_action(self, state, current_coord, ground_truth):
        """
        Select an action during the interaction with environment, using epsilon greedy policy
        This implementation should be used when training
        ----------
        Argument:
        state         - the state varible of current agent, consisting of (o,h), should conform to input shape of policy net
        current_coord - the current coordinate of this agent, should comes in four elements vector (left, right, top, bottom)
        ground_truth  - the groundtruth of current object
        ----------
        Return:
        An action index after conducting epsilon-greedy policy to current state
        """
        sample = random.random()
        # epsilon value is assigned by self.EPS
        eps_threshold = self.EPS
        # self.steps_done is to count how many steps the agent used to get final bounding box
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                    return action.cpu().numpy()[0]
                except:
                    return action.cpu().numpy()
        else:
            return self.get_best_next_action(current_coord, ground_truth)

    def select_action_model(self, state):
        """
        Select an action during the interaction with environment, using greedy policy
        This implementation should be used when testing
        ----------
        Argument:
        state - the state varible of current agent, consisting of (o,h), should conform to input shape of policy net
        ----------
        Return:
        An action index which is generated by policy net
        """
        with torch.no_grad():
            if use_cuda:
                inpu = Variable(state).cuda()
            else:
                inpu = Variable(state)
            qval = self.policy_net(inpu)
            _, predicted = torch.max(qval.data,1)
            #print("Predicted : "+str(qval.data))
            action = predicted[0] # + 1
            #print(action)
            return action
            
    def rewrap(self, coord):
        """
        A small function used to ensure every coordinate is inside [0,224]
        """
        return min(max(coord,0), 224)
    
    
    
    ########################
    # 3. Functions to form input tensor to policy network
    def get_features(self, image, dtype=FloatTensor):
        """
        Use feature extractor (a pre-trained CNN model) to transform an image to feature vectors
        ----------
        Argument:
        image          - an image representation, which should conform to the input shape of feature extractor network
        ----------
        Return:
        feature vector - a feature map which is another representation of the original image
                         dimension of this feature map depends on network, if using VGG16, it is (7,7,512)
        """
        # add first dimension to image and assign it to 1
        image = image.view(1,*image.shape)
        # change it to torch Variable
        image = Variable(image).type(dtype)
        if use_cuda:
            image = image.cuda()
        feature = self.feature_extractor(image)
        return feature.data
    
    def update_history(self, action):
        """
        Update action history vector with a new action
        ---------
        Argument:
        action         - a new taken action that should be updated into action history
        ---------
        Return:
        actions_history - a tensor of (9x9), encoding action history information
        """
        action_vector = torch.zeros(self.n_actions)
        action_vector[action] = 1
        for i in range(0,8,1):
            self.actions_history[i][:] = self.actions_history[i+1][:]
        self.actions_history[8][:] = action_vector[:]
        return self.actions_history
    
    def compose_state(self, image, dtype=FloatTensor):
        """
        Compose image feature and action history to a state variable
        ---------
        Argument:
        image - raw image data
        ---------
        state - a state variable, which is concatenation of image feature vector and action history vector
        """
        image_feature = self.get_features(image, dtype)
        image_feature = image_feature.view(1,-1)
        history_flatten = self.actions_history.view(1,-1).type(dtype)
        state = torch.cat((image_feature, history_flatten), 1)
        return state
    
    
    
    ########################
    # 4. Main training functions
    def optimize_model(self, verbose):
        """
        Sample a batch from experience memory and use this batch to optimize the model (DQN)
        """
        # if there are not enough memory, just do not do optimize
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        # every memory comes in foramt ('state', 'action', 'next_state', 'reward')
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # fetch next_state_batch, excluding final states
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states)).type(Tensor)
        # fetch state_batch
        state_batch = Variable(torch.cat(batch.state)).type(Tensor)
        if use_cuda:
            state_batch = state_batch.cuda()
        # fetch action_batch
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
        # fetch reward_batch
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)


        # use policy_net to generate q_values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # intialize state value for next states
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(Tensor)) 

        if use_cuda:
            non_final_next_states = non_final_next_states.cuda()
        
        # target_net is a frozen net that used to compute q-values, we do not update its weights
        with torch.no_grad():
            d = self.target_net(non_final_next_states) 
            next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)
        
        # compute expected q value
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        # compute loss
        loss = criterion(state_action_values, expected_state_action_values)
        
        if verbose:
            print("Loss:{}".format(loss))
            
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self, train_loader, verbose = False):
        """
        Use data in a train_loader to train an agent.
        This train_loader must contain images for only one class
        Each episode is done when this agent has interacted with all training images
        Each episode is performed as following:
        - Fetch a new training image
        - The agent take an action to interacte with this image using epsilon-greedy policy
          Each step will be pushed into experience replay
          After each step, update the weights of this network once
          The interaction finishes when triggered or up to 20 steps
        - Update the target net after the whole episode is done
        - Decrease epsilon
        - Save Network
        """
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        for i_episode in range(self.num_episodes):
            # Start i_episode
            print("Episode "+str(i_episode))
            img_id = 0
            # Traverse every training image to do interaction
            for key, value in  train_loader.items():
                
                if verbose:
                    img_id += 1
                    print("Training on Img {}/{}".format(img_id, len(train_loader.items())))
                    
                # fetch one image and ground_truth from train_loader
                image, ground_truth_boxes = extract(key, train_loader)
                original_image = image.clone()
                ground_truth = ground_truth_boxes[0]
                
                # initialization setting
                self.actions_history = torch.zeros((9,self.n_actions))
                new_image = image
                state = self.compose_state(image)
                
                original_coordinates = [xmin, xmax, ymin, ymax]
                self.current_coord = original_coordinates
                new_equivalent_coord = original_coordinates
              
                done = False
                t = 0
                
                # interaction with environment (image)
                while not done:
                    # increase step count
                    t += 1
                    
                    # take action according to epsilon-greedy policy
                    action = self.select_action(state, self.current_coord, ground_truth)
                    
                    # if action ==0, trigger
                    if action == 0:
                        next_state = None
                        closest_gt = self.get_max_bdbox(ground_truth_boxes, self.current_coord)
                        reward = self.compute_trigger_reward(self.current_coord, closest_gt)
                        done = True
                    
                    # if not, compute next coordinate
                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_position_box(self.current_coord, action)
                        new_xmin = self.rewrap(int(new_equivalent_coord[2])-16)
                        new_xmax = self.rewrap(int(new_equivalent_coord[3])+16)
                        new_ymin = self.rewrap(int(new_equivalent_coord[0])-16)
                        new_ymax = self.rewrap(int(new_equivalent_coord[1])+16)
                        
                        # fetch new_image (a crop of whole image) according to new coordinate
                        new_image = original_image[:, new_xmin:new_xmax, new_ymin:new_ymax]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        
                        
                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox(ground_truth_boxes, new_equivalent_coord)
                        reward = self.compute_reward(new_equivalent_coord, self.current_coord, closest_gt)
                        self.current_coord = new_equivalent_coord
                    
                    # tolerate
                    if t == 20:
                        done = True
                        
                    self.memory.push(state, int(action), next_state, reward)

                    # Move to the next state
                    state = next_state
                    image = new_image
                    
                    # Perform one step of the optimization (on the target network)
                    self.optimize_model(verbose)
                    
            
            # update target net every TARGET_UPDATE episodes
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # linearly decrease epsilon on first 5 episodes
            if i_episode < 5:
                self.EPS -= 0.18
                
            # Save network every episode
            self.save_network()

            print('Complete')
        
    def get_max_bdbox(self, ground_truth_boxes, actual_coordinates):
        """
        A simple function to hanlde more than 1 object in a picture
        It will compute IoU over every ground truth box and current coordinate and choose the largest one
        And return the corresponding ground truth box as actual ground truth
        """
        max_iou = False
        max_gt = []
        for gt in ground_truth_boxes:
            iou = self.intersection_over_union(actual_coordinates, gt)
            if max_iou == False or max_iou < iou:
                max_iou = iou
                max_gt = gt
        return max_gt
    
    
    
    
    ########################
    # 5. Predict and evaluate functions
    def predict_image(self, image, plot=False, verbose=False):
        """
        Run agent on a single image, taking actions until 40 steps or triggered
        The prediction process is following:
        - Initialization
        - Input state vector into policy net and get action
        - Take action and step into new state
        - Terminate if trigger or take up to 40 steps
        ----------
        Argument:
        image                - Input image, should be resized to (224,224) first
        plot                 - Bool variable, if True, plot all intermediate bounding box
        verbose              - Bool variable, if True, print out intermediate bouding box and taken action
        ---------
        Return:
        new_equivalent_coord - The final bounding box coordinates
        cross_flag           - If it should apply cross on the image, if done with trigger, True; if done with 40 steps, False
        steps                - how many steps it consumed
        """
        # set policy net to evaluation model, disable dropout
        self.policy_net.eval()
        
        # initialization
        original_image = image.clone()
        self.actions_history = torch.zeros((9,9))
        state = self.compose_state(image)
        
        new_image = image
        self.current_coord = [0, 224, 0, 224]
        steps = 0
        done = False
        cross_flag = True
        
        # start interaction
        while not done:
            steps += 1
            # take action according to greedy policy
            action = self.select_action_model(state)
            
            if action == 0:
                next_state = None
                new_equivalent_coord = self.current_coord
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_position_box(self.current_coord, action)
                
                new_xmin = self.rewrap(int(new_equivalent_coord[2])-16)
                new_xmax = self.rewrap(int(new_equivalent_coord[3])+16)
                new_ymin = self.rewrap(int(new_equivalent_coord[0])-16)
                new_ymax = self.rewrap(int(new_equivalent_coord[1])+16)
                
                new_image = original_image[:, new_xmin:new_xmax, new_ymin:new_ymax]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break            
                
                next_state = self.compose_state(new_image)
                self.current_coord = new_equivalent_coord
            
            if steps == 40:
                done = True
                cross_flag = False
            
            state = next_state
            image = new_image
            
            if verbose:
                print("Iteration:{} - Action:{} - Position:{}".format(steps, action, new_equivalent_coord))
            
            # if plot, print out current bounding box
            if plot:
                show_new_bdbox(original_image, new_equivalent_coord, color='b', count=steps)
                
            
        
        # if plot, save all changing in bounding boxes as a gif
#         if plot:
#             #images = []
#             tested = 0
#             while os.path.isfile('media/movie_'+str(tested)+'.gif'):
#                 tested += 1
#             # filepaths
#             fp_out = "media/movie_"+str(tested)+".gif"
#             images = []
#             for count in range(1, steps+1):
#                 images.append(imageio.imread(str(count)+".png"))
            
#             imageio.mimsave(fp_out, images)
            
#             for count in range(1, steps + 1):
#                 os.remove(str(count)+".png")
                
                
        return new_equivalent_coord, cross_flag, steps
    
    def predict_multiple_objects(self, image, plot=False, verbose=False):
        """
        Iteratively predict multiple objects, when one object is detected, draw a cross on it
        Perform up to 100 steps
        """
        
        new_image = image.clone()
        all_steps = 0
        bdboxes = []   
        
        while 1:
            bdbox, cross_flag, steps = self.predict_image(new_image, plot, verbose)
            bdboxes.append(bdbox)
            
            if cross_flag:
                mask = torch.ones((224,224))
                middle_x = round((bdbox[0] + bdbox[1])/2)
                middle_y = round((bdbox[2] + bdbox[3])/2)
                length_x = round((bdbox[1] - bdbox[0])/8)
                length_y = round((bdbox[3] - bdbox[2])/8)

                mask[middle_y-length_y:middle_y+length_y,int(bdbox[0]):int(bdbox[1])] = 0
                mask[int(bdbox[2]):int(bdbox[3]),middle_x-length_x:middle_x+length_x] = 0

                new_image *= mask
                
            all_steps += steps
                
            if all_steps > 100:
                break
                    
        return bdboxes
        
    
    def evaluate(self, dataset):
        """
        Conduct evaluation on a given dataset
        For each image in this dataset, using this agent to predict a bounding box on it
        Save predicted bdbox and ground truth bdbox to two lists
        Send these two lists to tool function eval_stats_at_threshold and get results
        *you can manually define threshold by setting threshold argument of this tool function*
        
        Return a dataframe that contains the result
        """
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in dataset.items():
            image, gt_boxes = extract(key, dataset)
            bbox = self.predict_multiple_objects(image)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)

        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        return stats

    
class Agent_3alpha(Agent):
  def __init__(self, classe, alpha=0.2, nu=3.0, threshold=0.5, num_episodes=15, load=False, model_name='vgg16', n_actions=25):
    super().__init__(classe=classe, alpha=alpha, nu=nu, threshold=threshold, num_episodes=num_episodes, load=load, model_name=model_name, n_actions=n_actions)


  def save_network(self):
      torch.save(self.policy_net, self.save_path + "_" + self.model_name + "_" +self.classe + '_3STEP')
      print('Saved')

  
  def calculate_position_box(self, current_coord, action):
    # action = [trigger,
    #           r_0.1, l_0.1, u_0.1, d_0.1, b_0.1, s_0.1, f_0.1, t_0.1,
    #           r_0.3, l_0.3, u_0.3, d_0.3, b_0.3, s_0.3, f_0.3, t_0.3,
    #           r_0.5, l_0.5, u_0.5, d_0.5, b_0.5, s_0.5, f_0.5, t_0.5] (24,)

    real_x_min, real_x_max, real_y_min, real_y_max = current_coord


    if (action-1)//8 == 0: alpha = self.alpha * 0.5
    elif (action-1)//8 == 1: alpha = self.alpha * 1.5
    elif (action-1)//8 == 2: alpha = self.alpha * 2.5
    else:
      alpha=self.alpha

    #print(action)

    alpha_h = alpha * (real_y_max - real_y_min)
    alpha_w = alpha * (real_x_max - real_x_min)

    if action%8 == 1: # Right 0.1 
        real_x_min += alpha_w * 0.5
        real_x_max += alpha_w * 0.5
    if action%8 == 2: # Left
        real_x_min -= alpha_w * 0.5
        real_x_max -= alpha_w * 0.5
    if action%8 == 3: # Up 
        real_y_min -= alpha_h * 0.5
        real_y_max -= alpha_h * 0.5
    if action%8 == 4: # Down
        real_y_min += alpha_h * 0.5
        real_y_max += alpha_h * 0.5
    if action%8 == 5: # Bigger
        real_y_min -= alpha_h * 0.5
        real_y_max += alpha_h * 0.5
        real_x_min -= alpha_w * 0.5
        real_x_max += alpha_w * 0.5
    if action%8 == 6: # Smaller
        real_y_min += alpha_h * 0.5
        real_y_max -= alpha_h * 0.5
        real_x_min += alpha_w * 0.5
        real_x_max -= alpha_w * 0.5
    if action%8 == 7: # Fatter
        real_y_min += alpha_h * 0.5
        real_y_max -= alpha_h * 0.5
    if (action%8 == 0) and (action != 0): # Taller
        real_x_min += alpha_w * 0.5
        real_x_max -= alpha_w * 0.5
            
    real_x_min = self.rewrap(real_x_min)
    real_x_max = self.rewrap(real_x_max)
    real_y_min = self.rewrap(real_y_min)
    real_y_max = self.rewrap(real_y_max)
    
    return [real_x_min, real_x_max, real_y_min, real_y_max]

#class Agent_PPO(Agent):

#class Agent 
