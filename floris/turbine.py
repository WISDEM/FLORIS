# Copyright 2017 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import griddata

class Turbine():
    """
    Turbine is model object representing a particular wind turbine. It is largely
    a container of data and parameters, but also contains method to probe properties
    for output.

    inputs:
        instance_dictionary: dict - the input dictionary as generated from the input_reader;
            it should have the following key-value pairs:
                {
        
                    "rotor_diameter": float,
        
                    "hub_height": float,
        
                    "blade_count": int,
        
                    "pP": float,
        
                    "pT": float,
        
                    "generator_efficiency": float,
        
                    "eta": float,
        
                    "power_thrust_table": dict,
        
                    "blade_pitch": float,
        
                    "yaw_angle": float,
        
                    "tilt_angle": float,
        
                    "TSR": float
        
                }

    outputs:
        self: Turbine - an instantiated Turbine object
    """

    def __init__(self, instance_dictionary):

        super().__init__()

        # constants
        self.grid_point_count = 16
        if np.sqrt(self.grid_point_count) % 1 != 0.0:
            raise ValueError("Turbine.grid_point_count must be the square of a number")

        self.velocities = [0] * self.grid_point_count
        self.grid = [0] * self.grid_point_count

        self.description = instance_dictionary["description"]

        properties = instance_dictionary["properties"]
        self.rotor_diameter = properties["rotor_diameter"]
        self.hub_height = properties["hub_height"]
        self.blade_count = properties["blade_count"]
        self.pP = properties["pP"]
        self.pT = properties["pT"]
        self.generator_efficiency = properties["generator_efficiency"]
        self.eta = properties["eta"]
        self.power_thrust_table = properties["power_thrust_table"]
        self.blade_pitch = properties["blade_pitch"]
        self.yaw_angle = properties["yaw_angle"]
        self.tilt_angle = properties["tilt_angle"]
        self.tsr = properties["TSR"]

        # these attributes need special attention
        self.rotor_radius = self.rotor_diameter / 2.0
        self.yaw_angle = np.radians(self.yaw_angle)
        self.tilt_angle = np.radians(self.tilt_angle)

        # initialize derived attributes
        self.fCp, self.fCt = self._CpCtWs()
        self.grid = self._create_swept_area_grid()
        # initialize to an invalid value until calculated
        self.velocities = [-1] * self.grid_point_count
        self.turbulence_intensity = -1
        self.plotting = False

        # calculated attributes are
        # self.Ct         # Thrust Coefficient
        # self.Cp         # Power Coefficient
        # self.power      # Power (W) <-- True?
        # self.aI         # Axial Induction
        # self.TI         # Turbulence intensity at rotor
        # self.windSpeed  # Windspeed at rotor

        # self.usePitch = usePitch
        # if usePitch:
        #     self.Cp, self.Ct, self.betaLims = CpCtpitchWs()
        # else:
        #     self.Cp, self.Ct = CpCtWs()

    # Private methods

    def _create_swept_area_grid(self):
        # TODO: add validity check:
        # rotor points has a minimum in order to always include points inside
        # the disk ... 2?
        #
        # the grid consists of the y,z coordinates of the discrete points which
        # lie within the rotor area: [(y1,z1), (y2,z2), ... , (yN, zN)]

        # update:
        # using all the grid point because that how roald did it.
        # are the points outside of the rotor disk used later?

        # determine the dimensions of the square grid
        num_points = int(np.round(np.sqrt(self.grid_point_count)))
        # syntax: np.linspace(min, max, n points)
        horizontal = np.linspace(-self.rotor_radius, self.rotor_radius, num_points)
        vertical = np.linspace(-self.rotor_radius, self.rotor_radius, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        # grid = [point for point in grid if np.hypot(point[0], point[1]) < self.rotor_radius]

        return grid

    def _calculate_cp(self):
        return self.fCp(self.get_average_velocity())

    def _calculate_ct(self):
        return self.fCt(self.get_average_velocity())

    def _calculate_power(self):
        cptmp = self.Cp \
                * np.cos(self.yaw_angle)**self.pP \
                * np.cos(self.tilt_angle)**self.pT
        return 0.5 * self.air_density * (np.pi * self.rotor_radius**2) \
                * cptmp * self.generator_efficiency \
                * self.get_average_velocity()**3

    def _calculate_ai(self):
        return 0.5 / np.cos(self.yaw_angle) \
               * (1 - np.sqrt(1 - self.Ct * np.cos(self.yaw_angle) ) )

    def _CpCtWs(self):
        cp = self.power_thrust_table["power"]
        ct = self.power_thrust_table["thrust"]
        windspeed = self.power_thrust_table["wind_speed"]

        fCpInterp = interp1d(windspeed, cp)
        fCtInterp = interp1d(windspeed, ct)

        def fCp(Ws):
            return max(cp) if Ws < min(windspeed) else fCpInterp(Ws)

        def fCt(Ws):
            return 0.99 if Ws < min(windspeed) else fCtInterp(Ws)

        return fCp, fCt

    def _calculate_swept_area_velocities(self, wind_direction, local_wind_speed, coord, x, y, z):
        """
            Initialize the turbine disk velocities used in the 3D model based on shear using the power log law.
        """
        u_at_turbine = local_wind_speed
        x_grid = x
        y_grid = y
        z_grid = z

        yPts = np.array([point[0] for point in self.grid])
        zPts = np.array([point[1] for point in self.grid])

        # interpolate from the flow field to get the flow field at the grid points
        dist = [np.sqrt( (coord.x - x_grid)**2 + (coord.y+yPts[i] - y_grid)**2 + (self.hub_height+zPts[i] - z_grid)**2  ) for i in range(len(yPts))]

        idx = [np.where(dist[i]==np.min(dist[i])) for i in range(len(yPts))]
        data = [u_at_turbine[idx[i]] for i in range(len(yPts))]

        return np.array(data)

    def _calculate_swept_area_velocities_visualization(self, grid_resolution, local_wind_speed, coord, x, y, z):

        dx = (np.max(x) - np.min(x)) / grid_resolution.x
        dy = (np.max(y) - np.min(y)) / grid_resolution.y
        mask = \
            (x <= coord.x + dx) & (x >= (coord.x - dx)) & \
            (y <= coord.y + dy) & (y >= coord.y - dy) & \
            (z < self.hub_height + self.rotor_radius) & (z > self.hub_height - self.rotor_radius)
        u_at_turbine = local_wind_speed[mask]
        x_grid = x[mask]
        y_grid = y[mask]
        z_grid = z[mask]
        data = np.zeros(len(self.grid))
        for i, point in enumerate(self.grid):
            data[i] = griddata(
                (x_grid, y_grid, z_grid),
                u_at_turbine,
                (coord.x, coord.y + point[0], self.hub_height + point[1]),
                method='nearest')
        return data

    # Public methods

    def calculate_turbulence_intensity(self, flowfield_ti, velocity_model, turbine_coord, wake_coord, turbine_wake):

        ti_initial = flowfield_ti

        # turbulence intensity parameters stored in floris.json
        ti_i = velocity_model.ti_initial
        ti_constant = velocity_model.ti_constant
        ti_ai = velocity_model.ti_ai
        ti_downstream = velocity_model.ti_downstream

        # turbulence intensity calculation based on Crespo et. al.
        ti_calculation = ti_constant \
                       * turbine_wake.aI**ti_ai \
                       * ti_initial**ti_i \
                       * ((turbine_coord.x - wake_coord.x) / self.rotor_diameter)**ti_downstream

        return np.sqrt(ti_calculation**2 + self.turbulence_intensity**2)

    # individual functions for loads
    def _diff_half(self):

        yPts = np.array([point[0] for point in self.grid])
        vel_half1 = self.velocities[yPts > 0]
        vel_half2 = self.velocities[yPts < 0]

        return (np.mean(vel_half2)-np.mean(vel_half1))


    def _spatial_std(self):
        return np.std(self.velocities)

    def _calculate_loads(self):#, area_overlap):

        # compute the loads based on:
        #   -turbulence intensity (rotor averaged turbulence intensity)
        #   -wake overlap (potentially gradient across wind turbine... horizontal shear, double integral, linear approximation, etc.)
        #   -wind speed (rotor averaged wind speed)
        #   -thrust coefficient (based on rotor averaged thrust)
        #   Future work: yaw misalignment

        # place holder for loads
        # since there aren't very many loads we are trying to estimate, all of them can be calculated at once
        loads = {}

        x0 = self.get_average_velocity()
        x1 = self.turbulence_intensity
        x2 = self._spatial_std()

        # Compute certain loads
        flap = 328.95436736 * x0 + 2302.19640733 *x2  -1312.3567263543023
        dt = 67.10114816 * x0 + 37.40466169 *x1  -420.8451807048049
        tow = -1032.52606333* x0 + 11978.24658772 *x2  + 10099.192383198206

        # define this above for a measure of asymmetry using self.velocities (the locations in self.grid)
        
        # x3 = self.Ct

        # individual functions for each load (right now, just a linear combination of average rotor velocity, ti, asymmetry, and Ct)
        #loads.DEL = dict()
        #loads.DEL['XXX'] = x0 + x1 + x2 + x3 
        #loads.DEL['XXX'] = x0 + x1 + x2 + x3 


        return flap, dt, tow

    def update_quantities(self, u_wake, coord, flowfield, rotated_x, rotated_y, rotated_z):

        # extract relevant quantities
        local_wind_speed = flowfield.initial_flowfield - u_wake

        # update turbine quantities
        if self.plotting:
            self.initial_velocities = self._calculate_swept_area_velocities_visualization(
                                        flowfield.grid_resolution,
                                        flowfield.initial_flowfield,
                                        coord,
                                        rotated_x,
                                        rotated_y,
                                        rotated_z)
            self.velocities = self._calculate_swept_area_velocities_visualization(
                flowfield.grid_resolution,
                                        local_wind_speed,
                                        coord,
                                        rotated_x,
                                        rotated_y,
                                        rotated_z)
        else:
            self.initial_velocities = self._calculate_swept_area_velocities(
                                        flowfield.wind_direction,
                                        flowfield.initial_flowfield,
                                        coord,
                                        rotated_x,
                                        rotated_y,
                                        rotated_z)
            self.velocities = self._calculate_swept_area_velocities(
                                        flowfield.wind_direction,
                                        local_wind_speed,
                                        coord,
                                        rotated_x,
                                        rotated_y,
                                        rotated_z)
        #self.Cp = self._calculate_cp()
        #self.Ct = self._calculate_ct()
        self.Cp = self._cp_pitch()
        self.Ct = self._ct_pitch()
        self.power = self._calculate_power()
        self.aI = self._calculate_ai()
        self.flap, self.dt, self.tow = self._calculate_loads()

    def set_yaw_angle(self, angle):
        """
        Sets the turbine yaw angle
        
        inputs:
            angle: float - new yaw angle in degrees
        
        outputs:
            none
        """
        self.yaw_angle = np.radians(angle)

    def get_average_velocity(self):
        return np.mean(self.velocities)

    def _ct_pitch(self):
        Ct = [0.820707172, 0.813335284, 0.811414582, 0.811926793, 0.811680909, 0.811583821, 0.811567464, 0.810653645, 0.768326153, 0.808381698, 0.890632335, 0.941170227, 0.973268287,
              0.789707172, 0.781335284, 0.780414582, 0.779926793, 0.779680909, 0.779883821, 0.779967464, 0.780053645, 0.741026153, 0.771881698, 0.856332335, 0.907470227, 0.941868287,
              0.756707172, 0.749335284, 0.747414582, 0.746926793, 0.747280909, 0.747283821, 0.747367464, 0.747353645, 0.713526153, 0.732481698, 0.816432335, 0.866670227, 0.901268287,
              0.723707172, 0.716335284, 0.714414582, 0.713926793, 0.713880909, 0.713983821, 0.714067464, 0.714153645, 0.685526153, 0.691381698, 0.772232335, 0.819770227, 0.851668287,
              0.689707172, 0.682335284, 0.680414582, 0.679926793, 0.680280909, 0.680283821, 0.680267464, 0.680453645, 0.657326153, 0.649381698, 0.725632335, 0.768470227, 0.795668287,
              0.655707172, 0.648335284, 0.647414582, 0.646726793, 0.646680909, 0.646583821, 0.646667464, 0.646653645, 0.629026153, 0.607981698, 0.678432335, 0.715370227, 0.736768287,
              0.621707172, 0.615335284, 0.613414582, 0.613026793, 0.612880909, 0.612883821, 0.612867464, 0.612953645, 0.600526153, 0.579181698, 0.632532335, 0.662970227, 0.678368287,
              0.586707172, 0.581335284, 0.579414582, 0.579526793, 0.579480909, 0.579483821, 0.579467464, 0.579353645, 0.572026153, 0.553481698, 0.588732335, 0.613170227, 0.622768287,
              0.552707172, 0.548335284, 0.546414582, 0.546526793, 0.546280909, 0.546383821, 0.546367464, 0.546453645, 0.542726153, 0.528081698, 0.548132335, 0.567070227, 0.571768287,
              0.519707172, 0.515335284, 0.514414582, 0.514226793, 0.514180909, 0.514083821, 0.514067464, 0.514053645, 0.513126153, 0.502381698, 0.510832335, 0.525170227, 0.525968287,
              0.487707172, 0.484335284, 0.483214582, 0.483126793, 0.482880909, 0.482883821, 0.482967464, 0.482953645, 0.482826153, 0.476181698, 0.476632335, 0.487670227, 0.485368287]


        Ct_n = np.array(Ct)
        Ct_n = Ct_n.reshape(11,13)

        wind_speed = np.array([3, 4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15])

        pitch = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

        f = interp2d(wind_speed,pitch,Ct_n)

        ct_val = f(self.get_average_velocity(),self.blade_pitch)

        return ct_val 


    def _cp_pitch(self):
        Cp = [0.4973,    0.4825,  0.4802,  0.4799,  0.4799,  0.48,    0.4802,  0.4802,  0.4723,  0.48,    0.4755, 0.456,   0.4324,
              0.4937,        0.4775,  0.4748,  0.4744,  0.4745,  0.4746,  0.4747,  0.4748,  0.466,   0.4735,  0.476,   0.4607,  0.4382,
              0.4876,        0.4701,  0.4671,  0.4667,  0.4667,  0.4668,  0.4669,  0.467,   0.4582,  0.4636,  0.4726,  0.4618,  0.442,
              0.4791,        0.4606,  0.4567,  0.4568,  0.4569,  0.4569,  0.4571,  0.4572,  0.4488,  0.4507,  0.4655,  0.4587,  0.4416,
              0.4684,        0.4493,  0.4458,  0.4453,  0.4453,  0.4453,  0.4454,  0.4455,  0.4381,  0.435,   0.4545,  0.4511,  0.4364,
              0.4556,        0.4364,  0.4328,  0.4321,  0.4321,  0.4322,  0.4323,  0.4324,  0.4261,  0.4173,  0.4403,  0.4394,  0.4266,
              0.441,         0.422,   0.4184,  0.4177,  0.4177,  0.4177,  0.4178,  0.4179,  0.4132,  0.4038,  0.4236,  0.4243,  0.4131,
              0.4249,        0.4063,  0.4027,  0.4021,  0.402,   0.4021,  0.4022,  0.4022,  0.3993,  0.3909,  0.4054,  0.4069,  0.3968,
              0.4076,       0.3897, 0.3862,  0.3855,  0.3855,  0.3855,  0.3856,  0.3857,  0.3843,  0.3774,  0.3864,  0.3884,  0.3788,
              0.3895,        0.3725,  0.3691,  0.3685,  0.3684,  0.3685,  0.3685,  0.3686,  0.3683,  0.3632,  0.3672,  0.3694,  0.3602,
              0.3707,        0.3548,  0.3516, 0.3511,  0.351,   0.351,   0.3511,  0.3512,  0.3512, 0.3482,  0.3484,  0.3506,  0.3418]

        Cp_n = np.array(Cp)
        Cp_n = Cp_n.reshape(11,13)

        wind_speed = np.array([3, 4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15])

        pitch = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

        f = interp2d(wind_speed,pitch,Cp_n)

        cp_val = f(self.get_average_velocity(),self.blade_pitch)

        return cp_val

    




    # coefficients for loads

