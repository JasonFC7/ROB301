#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray, UInt32
import numpy as np
import colorsys

class BayesLoc:
    def __init__(self, p0, colour_codes, colour_map, targets = []):
        self.colour_sub = rospy.Subscriber(
            "mean_img_rgb", Float64MultiArray, self.colour_callback, queue_size=1
        )
        self.line_sub = rospy.Subscriber("line_idx", UInt32, self.line_callback)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        self.num_states = len(p0)
        self.colour_codes = colour_codes
        self.colour_map = colour_map
        self.probabilities = p0
        self.state_prediction = np.zeros(self.num_states)

        self.counter = 0

        self.cur_colour = None  # most recent measured colour

        self.rate = rospy.Rate(30)
        self.int_err = 0
        self.kp = 1/320
        self.ki = 0.0015/320
        self.kd = 2/320
        self.errors = [0]  
        self.colour_vals = []
        self.calibrated = True
        self.calibrating_colour = None
        self.moving = True

        self.prediction_vals = []
        self.updated_prediction_vals = []

        self.state_model_matrix = np.array([[0.85, 0.05, 0.05],[0.1, 0.9, 0.1],[0.05, 0.05, 0.85]])
        self.measurement_model_matrix = np.array([[0.6, 0.2, 0.05, 0.05], [0.2, 0.6, 0.05,0.05], [0.05, 0.05, 0.65, 0.2], [0.05, 0.05, 0.15, 0.6], [0.1, 0.1, 0.1, 0.1]])

        self.targets = targets

    def colour_callback(self, msg):
        """
        callback function that receives the most recent colour measurement from the camera.
        """

        self.cur_colour = np.array(msg.data)  # [r, g, b]

        if self.calibrated == False:
            if len(self.colour_vals) < 100:

                self.colour_vals.append(self.cur_colour.tolist())
            else:
                self.calibrated = True
                self.colour_vals = np.array(self.colour_vals)
                avg_colour = [np.average(self.colour_vals[:, 0]),np.average(self.colour_vals[:, 1]),np.average(self.colour_vals[:, 2])]
                print("COLOUR VALUES FOR",self.calibrating_colour, avg_colour)
        
        
    def calibrate(self, colour):
        self.calibrated = False
        self.calibrating_colour = colour
        self.moving = False
        
        # rospy.loginfo(self.cur_colour)

    def line_callback(self, msg):
        if self.moving == True:
            vel = Twist()
            err = msg-320
            vel.linear.x = 0.1
            self.count = self.count + 1
            if err < 200:
                if type == 'bang': 
                    vel.linear.x = 0.15
                    if err > 0:
                        vel.angular.z = -0.2
                    elif err < 0:
                        vel.angular.z = 0.2
                    else:
                        vel.angular.z = 0
                elif type == 'P':
                    
                    vel.angular.z = -err*self.kp
                elif type == 'PI':
                    self.int_err = self.int_err+err
                    vel.angular.z = -err*self.kp - self.int_err*self.ki
                elif type == 'PID':
                    self.int_err = self.int_err+err
                    d_err1 = err - self.errors[-1] 
                    # if d_err < 10:
                    #     msg.linear.x = 0.35
                    vel.angular.z = -err*self.kp - self.int_err*self.ki - self.kd*d_err1 
                    if abs(vel.angular.z) > 1.5:
                        vel.linear.x = 0.05
                self.errors.append(err)
            else:
                vel.angular.z = 0
                vel.linear.x = 0.1
                self.errors.append(0)
        else:
            vel.linear.x = 0
            vel.angular.z = 0
            


        self.x_vel = vel.linear.x
        self.cmd_pub.publish(vel)
        self.rate.sleep()
        
    def wait_for_colour(self):
        """Loop until a colour is received."""
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and self.cur_colour is None:
            rate.sleep()

    def state_model(self, u):



        return np.dot(self.state_model_matrix[:, u+1], self.probability)


    def measurement_model(self, x):
        """
        Measurement model p(z_k | x_k = colour) - given the pixel intensity,
        what's the probability that of each possible colour z_k being observed?
        """
        eu_distances = []
        colour_prob = []
        for colour in self.colour_codes:
            eu_distances.append(np.linalg.norm(x - colour))
        eu_distances = np.array(eu_distances)
        for distance in eu_distances:
            colour_prob.append(distance/np.sum(eu_distances))


        
        if self.cur_colour is None:
            self.wait_for_colour()

        """
        TODO: You need to compute the probability of states. You should return a 1x5 np.array
        Hint: find the euclidean distance between the measured RGB values (self.cur_colour)
            and the reference RGB values of each colour (self.ColourCodes).
        """

        return colour_prob

    def state_predict(self, u):
        self.meas_colour = self.measurement_model.index(max(self.measurement_model))
        self.prediction = np.zeros(len(self.colour_map))
        if self.meas_colour == 4:
            self.prediction = self.probabilities
            pass
        else:
            for n in range (len(colour_map)):
                
                if n == len(colour_map)-2:
                    surrounding = self.probabilities[n-2:]
                elif n == len(colour_map)-1:
                    surrounding = self.probabilities[n-1:] + self.probabilities[0]
                else:
                    surrounding = self.probabilities[n-1:n+2]

                # p(k+1)|p(k) * p(k)+z(k)
                self.prediction[n] = self.state_model_matrix[0, u]*surrounding[2] + self.state_model_matrix[1, u]*surrounding[1] + self.state_model_matrix[2, u]*surrounding[2]
            if self.prediction.index(max(self.prediction)) in self.targets and self.counter < 30:
                self.counter += 1
                self.moving = False
            else:
                self.counter = 0
            self.state_update()

        # rospy.loginfo("predicting state")
        # rospy.loginfo(self.cur_colour)
        # rospy.loginfo(self.colour_sub.callback())
        """
        TODO: Complete the state prediction function: update
        self.state_prediction with the predicted probability of being at each
        state (office)
        """

    def state_update(self):
        rospy.loginfo("updating state")
        self.norm_factor = 0
        for n in range(len(colour_map)):
            self.probabilities[n] = self.prediction[n]*self.measurement_model_matrix[self.meas_colour, self.colour_map[n]]
            self.norm_factor += self.probabilities[n]

        self.probabilities = self.probabilities/self.norm_factor
        return self.probabilities 

        """
        TODO: Complete the state update function: update self.probabilities
        with the probability of being at each state
        """


if __name__ == "__main__":
    rospy.init_node("final_project")
    # This is the known map of offices by colour
    # 0: red, 1: green, 2: blue, 3: yellow, 4: line
    # current map starting at cell #2 and ending at cell #12
    colour_map = [3, 0, 1, 2, 2, 0, 1, 2, 3, 0, 1]

    # TODO calibrate these RGB values to recognize when you see a colour
    # NOTE: you may find it easier to compare colour readings using a different
    # colour system, such as HSV (hue, saturation, value). To convert RGB to
    # HSV, use:
    # h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    colour_codes = [
        [167, 146, 158],  # red
        [163, 184, 100],  # green
        [173, 166, 171],  # blue
        [167, 170, 117],  # yellow
        [150, 150, 150],  # line
    ]

    #CALIBRATE COLOUR VALUES:
    # PURPLE: 189.26270578125002, 138.29217390624999, 189.36151828125
    # GREEN: 149.909223046875, 183.724045859375, 143.19752929687502
    # YELLOW: 158.43043359375, 141.72061765625, 127.87986765625001
    # ORANGE: 243.88218390625, 141.095355703125, 101.69143664062501


    # initial probability of being at a given office is uniform
    p0 = np.ones_like(colour_map) / len(colour_map)

    localizer = BayesLoc(p0, colour_codes, colour_map)

    # rospy.spin()
    rospy.sleep(0.5)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        localizer.state_predict()

        rate.sleep()

    rospy.loginfo("finished!")
    rospy.loginfo(localizer.probability)
