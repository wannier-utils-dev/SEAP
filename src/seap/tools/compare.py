import os
import sys
import time

# This script compares the performance of different machine learning models for predicting
# Wannier functions from 3D data. It measures the time taken to run the deep learning model,
# Lasso Regression model, and Integration process.

# Handle both relative imports (when used as module) and absolute imports (when run directly)
try:
    from ..prediction.lasso import LassoRegression
    from ..prediction.integration import Integration
except ImportError:
    # Add the src directory to the path for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from seap.prediction.lasso import LassoRegression
    from seap.prediction.integration import Integration
# Note: deep_learning_model import is moved inside the function to avoid encoders dependency

def main():
    """
    Main function to compare the performance of different machine learning models.
    """
    # Set the number of features or dimensions for the models
    n = 4

    # Define the resolution of the 3D data
    nx = ny = nz = 32

    # Define file paths for the data and its metadata
    psi_name = '../image32x32x32.npy'
    psi_info = '../image_info.npy'

    # Create a dictionary to store data-related information
    psi_dict = {'name': psi_name, 'info': psi_info, 'resolution': [nx, ny, nz]}

    # Set the neural network identifier
    nnid = 1

    # Measure the time taken to run the deep learning model
    start_nn = time.time()
    try:
        # Handle both relative imports (when used as module) and absolute imports (when run directly)
        try:
            from ..core.predict import deep_learning_model
        except ImportError:
            from seap.core.predict import deep_learning_model
        deep_learning_model(n, nnid, **psi_dict)
    except ImportError as e:
        print(f"Deep learning model not available: {e}")
        print("Skipping neural network test")
    end_nn = time.time()
    print(f"time (nn) : {end_nn - start_nn}")

    # Set the number of training and prediction samples
    num_t = num_p = 64

    # Measure the time taken to run the Lasso Regression model
    start_lr = time.time()
    lr = LassoRegression(num_t, num_p, n, **psi_dict)
    r_lr, alpha, clm_lr = lr.run()
    end_lr = time.time()
    print(f"time (lasso) : {end_lr - start_lr}")

    # Measure the time taken to run the Integration process
    start_itg = time.time()
    itg = Integration(num_t, num_p, n, **psi_dict)
    r_itg, clm_itg = itg.run()
    end_itg = time.time()
    print(f"time (integration) : {end_itg - start_itg}")


if __name__ == '__main__':
    main()
