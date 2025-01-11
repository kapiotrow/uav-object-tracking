#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

class Tank(object):
    def __init__(self) -> None:
        self.vel_pub_topic = "/robotika_freyja_sensor_config_2/cmd_vel"
        self.vel_pub = rospy.Publisher(self.vel_pub_topic, Twist, queue_size=10)
        self.rate = rospy.Rate(20)
        

    def velocity_cb(self):
        pub_msg = Twist()
        pub_msg.linear.z = 0
        pub_msg.linear.y = 0
        pub_msg.linear.x = 0.3
        pub_msg.angular.z = 0.01
        self.vel_pub.publish(pub_msg)


if __name__ == "__main__":
    rospy.init_node("tank_control")
    control = Tank()
    print(rospy.is_shutdown())
    while(not rospy.is_shutdown()):
        control.velocity_cb()
        control.rate.sleep()