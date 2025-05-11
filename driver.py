import msgParser
import numpy as np
import time
import keyboard
import pandas as pd
import math
import carState
import carControl
import joblib

class Driver(object):
    def __init__(self, stage, mapnum=1):
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3

        self.stage = stage
        self.mapnum = mapnum
        self.lapNum = 0

        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.steer_lock = 0.785398
        self.max_speed = 300
        self.prev_rpm = None
        self.manual_steer = 0.0
        self.manual_accel = 0.0
        self.manual_brake = 0.0
        self.control.setGear(1)

        self.gear_reset_time = None
        self.check_rpm_time = False

        keyboard.on_release_key('k', self.gear_down)
        keyboard.on_release_key('l', self.gear_up)

        # Load the trained model and scaler
        self.model = joblib.load('mlp_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def init(self):
        self.angles = [0 for x in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        self.state.setFromMsg(msg)

        
        inp = [
            self.state.getAngle(),
            self.state.getGear(),
            self.state.getRpm(),
            self.state.getSpeedX(),
            self.state.getSpeedY(),
            self.state.getSpeedZ(),
            *self.state.getTrack()[:19],
            *self.state.getWheelSpinVel(),
            self.state.getZ()
        ]

        # Create DataFrame for model input
        featu = [
            'Angle', 'Gear', 'RPM', 'SpeedX', 'SpeedY', 'SpeedZ',
            'Track_1', 'Track_2', 'Track_3', 'Track_4', 'Track_5',
            'Track_6', 'Track_7', 'Track_8', 'Track_9', 'Track_10',
            'Track_11', 'Track_12', 'Track_13', 'Track_14', 'Track_15',
            'Track_16', 'Track_17', 'Track_18', 'Track_19',
            'WheelSpinVelocity_1', 'WheelSpinVelocity_2',
            'WheelSpinVelocity_3', 'WheelSpinVelocity_4', 'Z'
        ]
        df = pd.DataFrame([inp], columns=featu)

        
        scaled = self.scaler.transform(df) # Scale and predict
        out = self.model.predict(scaled)[0]

        accel = np.clip(out[0], 0, 1)
        brake = np.clip(out[1], 0, 1)
        clutch = out[2]
        gear = int(np.clip(out[4], -1, 6))

        
        postracking = self.state.getTrackPos() # Steering logic
        steer = 0.0
        if postracking < 0.1:
            steer = 0.05
        elif postracking > 0.9:
            steer = -0.05

        
        if self.state.getGear() <= 1: # Initial gear logic
            gear = 0
            accel = 0.3
            brake = 0.0

        
        rpm = self.state.getRpm() # Gear and RPM handling
        if rpm > 6000 and gear < 6:
            gear = min(gear + 1, 6)
            accel = 0.5
        elif rpm < 2000 and gear > 1:
            gear = max(gear - 1, 1)
            accel = 0.2

        
        self.control.setAccel(accel) # Apply control
        self.control.setBrake(0.0)
        self.control.setSteer(steer)
        self.control.setGear(gear)
        self.control.setClutch(clutch)

        return self.control.toMsg()

    def onShutDown(self):
        pass

    def onRestart(self):
        pass

    def gear_down(self, event):
        '''Decrease transmission gear'''
        self.control.setGear(max(-1, self.control.getGear() - 1))

    def gear_up(self, event):
        '''Increase transmission gear'''
        self.control.setGear(min(6, self.control.getGear() + 1))
