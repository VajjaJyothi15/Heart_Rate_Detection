import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyQualitySystem:
    """
    Evaluates signal quality and returns a confidence score from 0 to 100.
    """

    def __init__(self):

        self.noise = ctrl.Antecedent(
            np.arange(0, 1.1, 0.1),
            "noise"
        )

        self.motion = ctrl.Antecedent(
            np.arange(0, 1.1, 0.1),
            "motion"
        )

        self.conf = ctrl.Consequent(
            np.arange(0, 101, 1),
            "confidence"
        )

        self.noise["low"] = fuzz.trimf(self.noise.universe, [0, 0, 0.4])
        self.noise["high"] = fuzz.trimf(self.noise.universe, [0.3, 1, 1])

        self.motion["low"] = fuzz.trimf(self.motion.universe, [0, 0, 0.4])
        self.motion["high"] = fuzz.trimf(self.motion.universe, [0.3, 1, 1])

        self.conf["poor"] = fuzz.trimf(self.conf.universe, [0, 0, 40])
        self.conf["good"] = fuzz.trimf(self.conf.universe, [50, 80, 100])

        rules = [
            ctrl.Rule(self.noise["low"] & self.motion["low"], self.conf["good"]),
            ctrl.Rule(self.noise["high"] | self.motion["high"], self.conf["poor"])
        ]

        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

    def evaluate(self, signal):

        signal = np.array(signal, dtype=np.float32)

        if signal.size < 30:
            return 50.0

        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-6)

        recent = signal[-60:]

        kernel = np.ones(5, dtype=np.float32) / 5
        smooth = np.convolve(recent, kernel, mode="same")
        residual = recent - smooth

        noise = np.std(residual) / (np.std(recent) + 1e-6)
        motion = np.mean(np.abs(np.diff(recent))) / 2.0
        lighting = np.std(np.diff(recent, n=2)) / 4.0

        noise = float(np.clip(noise, 0, 1))
        motion = float(np.clip(motion, 0, 1))
        lighting = float(np.clip(lighting, 0, 1))

        self.sim.input["noise"] = noise
        self.sim.input["motion"] = motion

        self.sim.compute()

        confidence = float(self.sim.output["confidence"])
        confidence = confidence * (1.0 - 0.3 * lighting)

        return float(np.clip(confidence, 0, 100))
