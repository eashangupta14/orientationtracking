import numpy as np
import argparse
import os
from utils import dataload, model, panaroma

def main():
    # Main Function
    
    root_dir = f'../plots/{args.dataset}/{args.data_num}/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Load Data
    dataloader = dataload(args.dataset, args.data_num)
    dataloader.load()
    dataloader.print_data_stats()

    # Plot Data
    #dataloader.plot('imu')

    # Calibrateimu
    dataloader.calibrate_imu(root_dir, args.user)

    # Create Model
    motion_model = model()
    # Load Data Into model
    motion_model.load_data(dataloader.data)
    # Set Initial Value
    init_pos = [1,0,0,0]
    motion_model.set_init_position(init_pos)

    # Plot initial QT and initial angles
    #motion_model.plot_quaternion()
    #motion_model.plot_angles()
    #motion_model.plot_acceleration()
    
    # Optimisation
    motion_model.optimize(args.numiter, args.alpha, args.epsilon, args.num_count, root_dir)

    # Plot final QT
    motion_model.plot_quaternion(root_dir)
    motion_model.plot_angles(root_dir)
    motion_model.plot_acceleration(root_dir)
    
    
    #data = np.load('Data.npy', allow_pickle=True)
    #data = data.item()
    #optimised_quart = np.load('optimised.npy')
    
    if 'cam' in motion_model.data:
        # Panaroma Image
        processor = panaroma(motion_model.data,motion_model.optimised)
        # Step 1 Image to Spherical coordinates
        processor.img_to_sphere()
        print('Got camera Spherical Coordinates')
        # Step 2 Spherical to cartesian coorrdinates wrt camera
        processor.sphere_to_cartesian()
        print('Got camera Cartesion Coordinates')
        # Step 3 Camera to World Frame
        processor.img_to_world()
        print('Got world Cartesion Coordinates')
        # Step 4 World fram cartesian to spherical coordinates
        processor.cartesion_to_sphere()
        print('Got world spherical Coordinates')
        # Step 5 Spherical to world image
        processor.get_pixel_coord()
        print('Got Panaroma image coordinates')

        # make panaroma
        processor.stitch_pan(root_dir)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_num', type=int, default=1, help='Dataset Number')
    parser.add_argument('--dataset', type = str, default = 'train', help = 'Training dataset or test dataset')
    parser.add_argument('--alpha', type = float, default = 0.001, help = 'Optimisation Gain')
    parser.add_argument('--numiter', type = int, default = 100, help = 'Number of maximum iteration')
    parser.add_argument('--epsilon', type = float, default = 0.001, help = 'Minimum difference between cost of 2 iteration for convergance')
    parser.add_argument('--num_count', type = int, default = 10, help = 'Number of times difference between cost have to be less than epsilon')
    parser.add_argument('--user', action = 'store_true', help = 'Let user select points on the Graph for imu calibration.')

    args = parser.parse_args()
    main()
    