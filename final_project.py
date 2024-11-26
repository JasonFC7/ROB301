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
        self.prediction = p0
        self.state_prediction = np.zeros(self.num_states)
        self.stopping = False

        self.counter = 0

        self.cur_colour = None  # most recent measured colour

        self.rate = rospy.Rate(30)
        self.int_err = 0
        self.kp = 1/320
        self.ki = 0/320
        self.kd = 0/320
        self.errors = [0]  
        self.colour_vals = []
        self.calibrated = True
        self.calibrating_colour = None
        self.moving = 0 #0 is line following, 1 is stopped, 2 is moving in a straight line with no adjustment
        self.last_colour = None
        self.colour_counter = 0
        self.checked = False
        self.skip_counter = 0

        # self.prediction_vals = []
        # self.updated_prediction_vals = []

        self.colour_labels = ['purple', 'orange', 'red', 'brown', 'line']

        self.state_model_matrix = np.array([[0.85, 0.05, 0.05],[0.1, 0.9, 0.1],[0.05, 0.05, 0.85]])
        self.measurement_model_matrix = np.array([[0.6, 0.2, 0.05, 0.05], [0.2, 0.6, 0.05,0.05], [0.05, 0.05, 0.65, 0.2], [0.05, 0.05, 0.15, 0.6], [0.1, 0.1, 0.1, 0.1]])
        print(self.measurement_model_matrix[2,2])

        self.targets = targets

    def colour_callback(self, msg):
        """
        callback function that receives the most recent colour measurement from the camera.
        """

        self.cur_colour = np.array(msg.data)  # [r, g, b]

        if self.calibrated == False:
            self.moving = 1
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
        self.moving = 1
        
        # rospy.loginfo(self.cur_colour)

    def line_callback(self, msg):
        vel = Twist()
        if self.moving == 0:
            
            err = msg.data-320
            vel.linear.x = 0.05
            
            if self.colour_prob[4] > 0.075 and self.skip_counter < 75:
                self.skip_counter += 1

            else:
                self.skip_counter = 0
                if -err*self.kp != 0:
                    vel.angular.z = (-err*self.kp)/abs(-err*self.kp)*min(abs(err*self.kp), 0.5)*(1+self.skip_counter/40)

            if abs(vel.angular.z) > 0.25:
               vel.linear.x = 0.0375

        elif self.moving == 1:
            vel.linear.x = 0
            vel.angular.z = 0
        elif self.moving == 2:
            vel.linear.x = 0.05
            vel.angular.z = 0
            
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
        self.colour_prob = self.measurement_model(self.cur_colour)
        self.meas_colour = self.colour_prob.index(min(self.colour_prob))
        # print(self.colour_labels[self.meas_colour])
        self.prediction = np.zeros(len(self.colour_map))
        if self.meas_colour == 4:
            self.checked = False
            self.colour_counter =0
            self.moving = 0
            self.prediction = self.probabilities
        else:
            self.skip_counter = 0
            if (self.colour_counter > 20 and self.checked == False) or self.stopping == True:
                if self.stopping == False:
                    print("updating: {}".format(self.colour_labels[self.meas_colour]))
                    self.checked = True
                    for n in range (len(colour_map)):
                        if n == 0:
                            surrounding = np.array([self.probabilities[-1], self.probabilities[0], self.probabilities[1]])
                        elif n == len(colour_map)-2:
                            surrounding = self.probabilities[-3:]
                        elif n == len(colour_map)-1:
                            surrounding = np.array([self.probabilities[-2], self.probabilities[-1], self.probabilities[0]])
                        else:
                            surrounding = self.probabilities[n-1:n+2]
                        # p(k+1)|p(k) * p(k)+z(k)
                        self.prediction[n] = self.state_model_matrix[0, u]*surrounding[0] + self.state_model_matrix[1, u]*surrounding[1] + self.state_model_matrix[2, u]*surrounding[2]
                    self.state_update()

                if self.probabilities.tolist().index(max(self.probabilities)) in self.targets and self.counter < 30:
                    self.counter += 1
                    self.stopping = True
                    self.moving = 1
                else:
                    self.stopping = False
                    self.counter = 0
                    self.moving = 2
            elif self.last_colour == self.meas_colour:
                self.colour_counter += 1
            elif self.last_colour != self.meas_colour:
                self.last_colour = self.meas_colour
                self.colour_counter = 0
                
        

        # rospy.loginfo("predicting state")
        # rospy.loginfo(self.cur_colour)
        # rospy.loginfo(self.colour_sub.callback())
        """
        TODO: Complete the state prediction function: update
        self.state_prediction with the predicted probability of being at each
        state (office)
        """

    def state_update(self):
        # rospy.loginfo("updating state")
        self.norm_factor = 0
        for n in range(len(colour_map)):
            self.probabilities[n] = self.prediction[n]*self.measurement_model_matrix[self.meas_colour, self.colour_map[n]]
            self.norm_factor += self.probabilities[n]

        self.probabilities = self.probabilities/self.norm_factor
        print(self.probabilities)
        if max(self.probabilities) > 0.5:
                    # pass
            print("At", self.prediction.tolist().index(max(self.prediction)), "with", max(self.prediction), "probability.")
        return self.probabilities 

        """
        TODO: Complete the state update function: update self.probabilities
        with the probability of being at each state
        """


if __name__ == "__main__":
    rospy.init_node("final_project")
    # This is the known map of offices by colour
    # 0: purple, 1: orange, 2: red, 3: brown, 4: line
    # current map starting at cell #2 and ending at cell #12
    colour_map = [2, 0, 1, 2, 0, 1, 3, 3, 0, 1, 3]

    # TODO calibrate these RGB values to recognize when you see a colour
    # NOTE: you may find it easier to compare colour readings using a different
    # colour system, such as HSV (hue, saturation, value). To convert RGB to
    # HSV, use:
    # h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    colour_codes = [
        [170.886418828125, 120.84823132812502, 183.833909140625],  # purple
        [238.06590539062498, 130.015980703125, 97.94329164062499],  # orange
        [224.50469070312502, 65.34891476562498, 110.694511015625],  # red
        [219.77502273437506, 136.64130742187498, 111.318619609375],  # brown
        [147.653547421875, 135.294512109375, 140.608622109375],  # line
    ]

    #CALIBRATE COLOUR VALUES:
    # PURPLE: 189.26270578125002, 138.29217390624999, 189.36151828125
    # GREEN: 149.909223046875, 183.724045859375, 143.19752929687502
    # YELLOW: 158.43043359375, 141.72061765625, 127.87986765625001
    # ORANGE: 238.06590539062498, 130.015980703125, 97.94329164062499
    # RED: 225.67106742187502, 66.26505210937499, 111.897434921875
    # BROWN: 209.68690078124996, 123.36574546875, 105.46604890625
    # WHITE LINE: 147.653547421875, 135.294512109375, 140.608622109375


    # initial probability of being at a given office is uniform
    p0 = np.ones_like(colour_map) / len(colour_map)

    localizer = BayesLoc(p0, colour_codes, colour_map, [5, 8, 10])

    # rospy.spin()
    rospy.sleep(0.5)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        localizer.state_predict(1)
        # localizer.calibrate()

        rate.sleep()

    rospy.loginfo("finished!")
    rospy.loginfo(localizer.probabilities)
