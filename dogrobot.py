import snowboydecoder
import rospy
import numpy as np
import time
import sys
import signal
import threading
import cv2
import time
import freenect
from playsound import playsound
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from math import radians, degrees
from geometry_msgs.msg import *




interrupted = False
from geometry_msgs.msg import Twist
def signal_handler(signal, frame):
    global interrupted
    interrupted = True


def interrupt_callback():
    global interrupted
    return interrupted

class GoForward():
    def __init__(self):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)

        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)
        
	# Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
     
	#TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10);
        # Twist is a datatype for velocity
        global move_cmd
        # Twist is a datatype for velocity
        move_cmd = Twist()
	# let's go forward at 0.2 m/s
        move_cmd.linear.x = 0
	# let's turn at 0 radians/s
        move_cmd.angular.z = 0


	# as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
	    # publish the velocity
            self.cmd_vel.publish(move_cmd)
	    # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()
    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
	# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
	# sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep()

def detected_right_callback():
    print "right detected"
    move_cmd.linear.x = 0
    move_cmd.angular.z = -np.pi/2
    time.sleep(1.4)
    move_cmd.linear.x = 0.2
    move_cmd.angular.z = 0
    global stopp
    stopp = 0

def detected_left_callback():
    print "left detected"
    move_cmd.linear.x = 0
    move_cmd.angular.z = np.pi/2
    time.sleep(1.4)
    move_cmd.linear.x = 0.2
    move_cmd.angular.z = 0
    global stopp
    stopp = 0

def detected_stop_callback():
    print "stop detected"
    move_cmd.linear.x = 0
    move_cmd.angular.z = 0
    global stopp
    stopp = 1

def detected_forward_callback():
    print "forward detected"
    global collision
    if not collision:
        move_cmd.linear.x = 0.2
        move_cmd.angular.z = 0
        global stopp
        stopp = 0
    else:
       print "not gonna move!!"


def create_nav_goal(x, y, yaw):
    """Create a MoveBaseGoal with x, y position and yaw rotation (in degrees).
Returns a MoveBaseGoal"""
    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header.frame_id = '/map' # Note: the frame_id must be map
    mb_goal.target_pose.pose.position.x = x
    mb_goal.target_pose.pose.position.y = y
    mb_goal.target_pose.pose.position.z = 0.0 # z must be 0.0 (no height in the map)

    # Orientation of the robot is expressed in the yaw value of euler angles
    angle = radians(yaw) # angles are expressed in radians
    quat = quaternion_from_euler(0.0, 0.0, angle) # roll, pitch, yaw
    mb_goal.target_pose.pose.orientation = Quaternion(*quat.tolist())

    return mb_goal

def callback_pose(data):
    """Callback for the topic subscriber.
Prints the current received data on the topic."""
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    roll, pitch, yaw = euler_from_quaternion([data.pose.pose.orientation.x,
                                             data.pose.pose.orientation.y,
                                             data.pose.pose.orientation.z,
                                             data.pose.pose.orientation.w])
    rospy.loginfo("Current robot pose: x=" + str(x) + "y=" + str(y) + " yaw=" + str(degrees(yaw)) + "o")

            
def detected_shafa_callback():
    print "shafa detected"


    # Connect to the navigation action server
    nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    rospy.loginfo("Connecting to /move_base AS...")
    nav_as.wait_for_server()
    rospy.loginfo("Connected.")

    rospy.loginfo("Creating navigation goal...")
    nav_goal = create_nav_goal(6.11104,1.90293, 0.0) 
    # 0.3003,-0.4244 left middle -1.7117,0.2116 TV -5.21498,1.98011 B526 -21.08758,20.04571 elevator 5.41223,-22.96190 girl toilet  8.05335,-27.73990 boy toilet
    rospy.loginfo("Sending goal")
    nav_as.send_goal(nav_goal)
    rospy.loginfo("Waiting for result...")
    nav_as.wait_for_result()
    nav_res = nav_as.get_result()
    nav_state = nav_as.get_state()
    rospy.loginfo("Done!")
    print "Result: ", str(nav_res) # always empty, be careful
    print "Nav state: ", str(nav_state) # use this, 3 is SUCCESS, 4 is ABORTED (couldnt get there), 5 REJECTED (the goal is not attainable)

def detected_boy_callback():
    print "boy detected"


    # Connect to the navigation action server
    nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    rospy.loginfo("Connecting to /move_base AS...")
    nav_as.wait_for_server()
    rospy.loginfo("Connected.")

    rospy.loginfo("Creating navigation goal...")
    nav_goal = create_nav_goal(8.05335,-27.73990, 0.0) 
    # 0.3003,-0.4244 left middle -1.7117,0.2116 TV -5.21498,1.98011 B526 -21.08758,20.04571 elevator 5.41223,-22.96190 girl toilet  8.05335,-27.73990 boy toilet
    rospy.loginfo("Sending goal")
    nav_as.send_goal(nav_goal)
    rospy.loginfo("Waiting for result...")
    nav_as.wait_for_result()
    nav_res = nav_as.get_result()
    nav_state = nav_as.get_state()
    rospy.loginfo("Done!")
    print "Result: ", str(nav_res) # always empty, be careful
    print "Nav state: ", str(nav_state) # use this, 3 is SUCCESS, 4 is ABORTED (couldnt get there), 5 REJECTED (the goal is not attainable)

def detected_105_callback():
    print "105 detected"


    # Connect to the navigation action server
    nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    rospy.loginfo("Connecting to /move_base AS...")
    nav_as.wait_for_server()
    rospy.loginfo("Connected.")

    rospy.loginfo("Creating navigation goal...")
    nav_goal = create_nav_goal(5.41223,-22.96190, 0.0) 
    # 0.3003,-0.4244 left middle -1.7117,0.2116 TV -5.21498,1.98011 B526 -21.08758,20.04571 elevator 5.41223,-22.96190 girl toilet  8.05335,-27.73990 boy toilet
    rospy.loginfo("Sending goal")
    nav_as.send_goal(nav_goal)
    rospy.loginfo("Waiting for result...")
    nav_as.wait_for_result()
    nav_res = nav_as.get_result()
    nav_state = nav_as.get_state()
    rospy.loginfo("Done!")
    print "Result: ", str(nav_res) # always empty, be careful
    print "Nav state: ", str(nav_state) # use this, 3 is SUCCESS, 4 is ABORTED (couldnt get there), 5 REJECTED (the goal is not attainable)


models = ['snowboy_model/right2.pmdl', 'snowboy_model/left2.pmdl', 'snowboy_model/stop2.pmdl', 'snowboy_model/forward2.pmdl']
models2 = ['snowboy_model/shafa.pmdl']
signal.signal(signal.SIGINT, signal_handler)

sensitivity = [0.4]*len(models)
sensitivity2 = [0.4]*len(models2)
detector = snowboydecoder.HotwordDetector(models, sensitivity=sensitivity)
detector2 = snowboydecoder.HotwordDetector(models2, sensitivity=sensitivity2)
callbacks = [detected_right_callback, detected_left_callback, detected_stop_callback, detected_forward_callback]
callbacks2 = [detected_shafa_callback]

def forward():
    GoForward()
def det():
    print "Start listening"
    detector.start(detected_callback=callbacks, interrupt_check=interrupt_callback, sleep_time=0.03)
def det2():
    print "Start listening"
    detector2.start(detected_callback=callbacks2, interrupt_check=interrupt_callback, sleep_time=0.03)

def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

def red_light_det():
    print "Start red light det"
    red_light_cascade = cv2.CascadeClassifier('opencv_haarcascade/red-light-cascade-stage20.xml')
    number_10_cascade = cv2.CascadeClassifier('opencv_haarcascade/number-10-cascade-stage20.xml')
    number_8_cascade = cv2.CascadeClassifier('opencv_haarcascade/number-8-cascade-stage20.xml')
    global stopp
    now_seconds_count = time.clock()
    red_seconds = 0
    countdowntime = 6
    diff = 0
    redd = 0
    temp = 0
    reded = 0
    sized = 0
    saw = 0
    playedd = 0
    while 1:
        img = get_video()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        red_light = red_light_cascade.detectMultiScale(gray, 1.3, 5) #detect red traffic light
        number_10 = number_10_cascade.detectMultiScale(gray, 1.3, 15)
        number_8 = number_8_cascade.detectMultiScale(gray, 1.3, 15)
        if red_light == ():
            now_seconds_count = time.clock()
            time.ctime(now_seconds_count)
            diff = int(now_seconds_count) - int(red_seconds)
            if (redd == 0) or (diff >= countdowntime):
                redd = 0
                font = cv2.FONT_HERSHEY_SIMPLEX
		if reded == 1:
		    playsound("voice/straight.mp3")
		    move_cmd.linear.x = 0.2
                    move_cmd.angular.z = 0
		    reded = 0
                    stopp = 0
                cv2.putText(img,'not red',(20,30), font, 1,(0,255,255),2,cv2.LINE_AA)
                playedd = 0
            elif (diff < countdowntime) and (redd == 1):
            	playedd = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                if temp != countdowntime - diff - 1:
                    print('countdown:',countdowntime - diff - 1)
                temp = countdowntime - diff - 1
                cv2.putText(img,'red(waiting for countdown)',(20,30), font, 1,(0,0,255),2,cv2.LINE_AA)
                reded = 1
                move_cmd.linear.x = 0
                move_cmd.angular.z = 0
                
        else:
            redd = 1
            diff = 0
	    reded = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            move_cmd.linear.x = 0
            move_cmd.angular.z = 0
            stopp = 1
            if not playedd:
            	playsound('voice/red_now2.mp3')
            	playedd = 1
            cv2.putText(img,'red',(20,30), font, 1,(0,0,255),2,cv2.LINE_AA)
            move_cmd.linear.x = 0
            move_cmd.angular.z = 0
            red_seconds = time.clock()
            time.ctime(red_seconds)
       # if (not number_10 == () or not number_8 == ()) and not saw:
       #     saw = 1
       #     move_cmd.linear.x = 0
       #     move_cmd.angular.z = 0
       #     playsound('wait_green.mp3')
        for (x,y,w,h) in red_light:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
       # for (x,y,w,h) in number_10:
       #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


def get_depth():
    depthh = freenect.sync_get_depth()[0]
    np.clip(depthh, 0, 2**10 - 1, depthh)
    depthh >>= 2
    depthh = depthh.astype(np.uint8)
    return depthh


def collision_det():
    print "Start collision_det"
    global collision
    collision = 0
    printed = 0
    while 1:
        depthh = get_depth()
        count = 0
        countt = 0
        global stopp
        for i in range(120,350, 5):
            for j in range(0,640, 5):
                if (depthh[i][j] < 180 and not depthh[i][j] == 255):
                    count += 1;
                elif depthh[i][j] == 255: 
                    countt += 1
        #print "count = ", count
        #print "countt = ", countt
        #print "stopp = ", stopp 
        #print "collision = ", collision
        if count > 5888*0.2 or countt > 5888*0.9:
            if collision == 0:
                print("object in front stop now!!")
                collision = 1
                move_cmd.linear.x = 0
                move_cmd.angular.z = 0
                time.sleep(1.2)
        elif collision == 1 and stopp == 0:
            collision = 0
            print "collisioned"
            if move_cmd.angular.z == 0:
                move_cmd.linear.x = 0.2
                move_cmd.angular.z = 0  
        elif collision == 1 and stopp == 1:
            collision = 0

def gen_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size = 10)
    #rospy.init_node('initial_pose'); #, log_level=roslib.msg.Log.INFO)
    rospy.loginfo("Setting Pose")
    rate = rospy.Rate(1)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback_pose)
    pose = geometry_msgs.msg.PoseWithCovarianceStamped()
    pose.header.frame_id = "map"
    pose.pose.pose.position.x= 3.55347 #0.21596
    pose.pose.pose.position.y= -1.34292 #0.73950
    pose.pose.pose.position.z=0
    pose.pose.covariance=[0.013950132964145823, 0.0019072991840314302, 0.0, 0.0, 0.0, 0.0, 0.0019072991840314302, 0.007761956238762302, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0030076962783880546]
    pose.pose.pose.orientation.z=-0.02529 #-0.96914
    pose.pose.pose.orientation.w=0.99968 #0.24647 
    rospy.loginfo(pose)
    pub.publish(pose)
    rate.sleep()
    pub.publish(pose)
    rate.sleep()
    pub.publish(pose)
    rate.sleep()


if __name__ == '__main__':
    global stopp
    stopp = 1
    mode = input("please enter mode 1 or 2: ")
    thread = threading.Thread(target=det)
    thread2 = threading.Thread(target=det2)
    thread_cam = threading.Thread(target=red_light_det)
    thread_depth = threading.Thread(target=collision_det)
    if int(mode) == 1:
        thread.start()
        thread_cam.start()
        thread_depth.start()
        forward()
    elif int(mode) == 2:
        thread2.start()
        rospy.init_node("navigation_mode")
        gen_pose()      
        



