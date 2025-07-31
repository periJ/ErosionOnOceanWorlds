##################################################################################################
#Module to contain all functions for my ocean world erosion project!
##################################################################################################
import numpy as np
import matplotlib.pyplot as plt


###################################################################################################
def settling_velocity(diameter,density_grain,density_fluid,kinematic_viscosity,grav_acc,CSF,powers):
    """Following Dietrich 1982, calculates the fall velocity of a sediment grain through a fluid"""
    #parameters
    #diameter = diameter of grain in m
    #density_grain = density of sediment grain in kg/m^3
    #density_fluid = density of fluid in kg/m^3
    #kinematic_viscosity = kinematic (NOT dynamic) viscosity in m^2/s
    #grav_acc = gravitational acceleration, in m/s^2
    #CSF = corey shape function (see Dietrich 1982)
    #powers = powers roundness index (see Dietrich 1982)
    
    #constants
    dimensionless_diameter = (density_grain - density_fluid)*grav_acc*diameter**3 / (density_fluid*kinematic_viscosity**2)
    
    
    #R1
    R1 =  -3.76715 + 1.92944*np.log10(dimensionless_diameter) - 0.09815*(np.log10(dimensionless_diameter))**2 - 0.00575*(np.log10(dimensionless_diameter))**3 + 0.00056*(np.log10(dimensionless_diameter))**4    
    
    #R2
    R2 = np.log10(1-((1-CSF)/0.85)) - (1-CSF)**2.3*np.tanh(np.log10(dimensionless_diameter)-4.6) + 0.3*(0.5-CSF)*(1-CSF)**2*(np.log10(dimensionless_diameter) - 4.6)
    
    #R3
    R3 = (0.65 - (CSF/2.85*np.tanh(np.log10(dimensionless_diameter)-4.6)))**(1+((3.5-powers)/2.5))
    
    #settling velocity
    dimensionless_velocity = R3*10**(R1+R2)
    settling_velocity = (dimensionless_velocity*(density_grain-density_fluid)*grav_acc*kinematic_viscosity / density_fluid)**(1/3)
    
    return settling_velocity #in m/s
    
###################################################################################################
def calc_hydraulic_radius(discharge, channel_width, grav_acc, slope, material, special_number):
    """Calculate the hydraulic radius of a given flow, using Manning (see SD04) or Darcy-Weisbach (see Wilson+2004) eqns"""
	#parameters
	#discharge = fluid discharge of the channel, in m^3/s
	#channel width = in m
	#grav_acc = gravitational acceleration, in m/s^2
	#slope = slope of the channel
	#material = gravel, boulder, sand for Darcy Weisbach, or Manning for Manning
	#special number = D50 or D84 for Darcy-Weisbach, roughness for Manning
	
    if material == 'gravel':
        
        D84 = special_number
        hydraul_radius = np.zeros_like(discharge,dtype = float)
        rh_test = np.arange(0.01,5,0.001)
        left = channel_width*rh_test / (channel_width - 2*rh_test)
        for i in range(len(discharge)):
            fc_over_8 = (5.75*np.log10(rh_test / D84)+3.514)**-2
            right = np.sqrt(discharge[i]**2 / (rh_test*channel_width**2*grav_acc*slope)*fc_over_8)
            idx = np.argmin(abs(left-right))
            hydraul_radius[i] = rh_test[idx]#*discharge[i]/discharge[-1]*3
            #print(idx,left[idx],right[idx])
           
    elif material == 'boulder':
    
        D84 = special_number
        hydraul_radius = np.zeros_like(discharge,dtype = float)
        rh_test = np.arange(0.01,5,0.001)
        left = channel_width*rh_test / (channel_width - 2*rh_test)
        for i in range(len(discharge)):
            fc_over_8 = (5.62*np.log10(rh_test / D84)+4)**(-2)
            right = np.sqrt(discharge[i]**2 / (rh_test*channel_width**2*grav_acc*slope)*fc_over_8)
            hydraul_radius[i] = rh_test[np.argmin(abs(left-right))]
    
    elif material == 'sand':
    
        D50 = special_number
        hydraul_radius = np.zeros_like(discharge)
        Hw_test = np.arange(0.01,100,0.001)
        left = Hw_test*channel_width / (2*Hw_test+channel_width)
        for i in range(len(discharge)):
            #right = (discharge[i]**2*D50**2.01 / (grav_acc*slope*Hw_test**2*channel_width**2))**(1/3.01) #using Darcy-Weisbach instead of Manning WRONG EQUATION BEACUSE IM DUMB! USED THIS UNTIL 1/22/25 WHOOPS
            right = (discharge[i]*D50**0.1005 / (8.46*np.sqrt(grav_acc*slope)*Hw_test*channel_width))**(1/0.6005)
            flow_depth = Hw_test[np.argmin(abs(left-right))]
            hydraul_radius[i] = flow_depth*channel_width / (2*flow_depth+channel_width)
    
    elif material == 'manning' or material == 'Manning':
    
        roughness = special_number
        hydraul_radius = np.zeros_like(discharge)
        Hw_test = np.arange(0.01,10,0.001)
        left = Hw_test*channel_width / (2*Hw_test+channel_width)
        for i in range(len(discharge)):
            right = (discharge[i]*roughness / (Hw_test*channel_width*np.sqrt(slope)))**(3/2)
            flow_depth = Hw_test[np.argmin(abs(left-right))]
            hydraul_radius[i] = flow_depth*channel_width / (2*flow_depth+channel_width) 

    elif material == 'Geoff' or 'geoff':
        roughness = special_number
        hydraul_radius = np.zeros_like(discharge)
        rh_test = np.arange(0.01,10,0.0001)
        left = channel_width*rh_test / (channel_width - 2*rh_test)
        for i in range(len(discharge)):
            right = discharge[i] / channel_width / (5.75*np.log10(12.2*rh_test / roughness)* np.sqrt(grav_acc*rh_test*slope))
            hydraul_radius[i] = rh_test[np.argmin(abs(left-right))]
           
            
    return hydraul_radius    #in m    
    
################################################################################################
def calc_hydraulic_radius_with_meanflowspeed(mean_flow_speed, grav_acc, slope, material, special_number):
    """Calculate the hydraulic radius of a given flow, using Manning (see SD04) or Darcy-Weisbach (see Wilson+2004) eqns"""
	#parameters
	#mean_flow_speed = mean fluid flow speed [m/s]
	#channel width = in m
	#grav_acc = gravitational acceleration, in m/s^2
	#slope = slope of the channel
	#material = gravel, boulder, sand for Darcy Weisbach, or Manning for Manning
	#special number = D50 or D84 for Darcy-Weisbach, roughness for Manning, grain_size for "choose for me"
	
    if material in 'choose for me':
        grain_size = special_number
        if grain_size >= 6.3e-5 and grain_size < 0.002: #between 0.063 and 2 mm = sand
            material = 'sand'
            print(material)
        elif grain_size >= 0.002 and grain_size < 0.256: #between 2 and 256 mm  = gravel
            material = 'gravel'
        elif grain_size >= 0.256:
            material = 'boulder'
        else:
            print('Grain Size too small!')
            return -1
    else:
        1
    #print(material)
	
    if material in 'gravel' or material in "Gravel":
        D84 = special_number
        hydraul_radius = np.zeros_like(mean_flow_speed,dtype = float) 
        rh_test = np.arange(0.0001,5,0.0001)
        fc_over_8_test = (5.75*np.log10(rh_test / D84)+3.514)**-2
        uw_test = np.sqrt(grav_acc*rh_test*slope)/np.sqrt(fc_over_8_test)
        for i in range(len(mean_flow_speed)):
            idx = np.argmin(abs(uw_test - mean_flow_speed[i]))
            hydraul_radius[i] = rh_test[idx]
           
    elif material in 'boulder':
        D84 = special_number
        hydraul_radius = np.zeros_like(mean_flow_speed,dtype = float)
        rh_test = np.arange(0.0001,5,0.0001)
        fc_over_8_test = (5.62*np.log10(rh_test / D84)+4)**(-2)
        uw_test = np.sqrt(grav_acc*rh_test*slope)/np.sqrt(fc_over_8_test)
        for i in range(len(mean_flow_speed)):
        	idx = np.argmin(abs(uw_test - mean_flow_speed[i]))
        	hydraul_radius[i] = rh_test[idx]
    
    elif material in 'sand':
    	D50 = special_number
    	hydraul_radius = np.zeros_like(mean_flow_speed,dtype = float)
    	for i in range(len(mean_flow_speed)):
    	    hydraul_radius[i] = (mean_flow_speed[i]*D50**0.1005 / (8.46*np.sqrt(grav_acc*slope)))**(1/0.6005)
    
    elif material in 'manning' or material in 'Manning':
    
        roughness = special_number
        hydraul_radius = np.zeros_like(mean_flow_speed,dtype = float)
        for i in range(len(mean_flow_speed)):
        	hydraul_radius[i] = (mean_flow_speed*roughness / np.sqrt(slope))**(3/2)

    elif material in 'Geoff' or 'geoff':
        roughness = special_number
        hydraul_radius = np.zeros_like(mean_flow_speed,dtype = float)
        rh_test = np.arange(0.001,5,0.001)
        fc_over_8_test = (5.75*np.log10(12.2*rh_test / roughness))**-2
        uw_test = np.sqrt(grav_acc*rh_test*slope)/np.sqrt(fc_over_8_test)
        for i in range(len(mean_flow_speed)):
        	idx = np.argmin(abs(uw_test - mean_flow_speed[i]))
        	hydraul_radius[i] = rh_test[idx]
           
            
    return hydraul_radius    #in m        

###################################################################################################
def spalding_shear_stress(mean_flow_vel,density,kinematic_viscosity,height_above_bed_dimless):
    """Use Eqn. 9 from Spalding 1961 to calculate shear stress along the bed for a given flow"""
    
    dynamic_viscosity = kinematic_viscosity*density
    
    #test lots of shear stress values:
    shear_stress = np.logspace(-6,3,1e6)
    
    #for each test value, calc both sides of Eqn. 9 in Spalding 
    height_above_bed = height_above_bed_dimless*dynamic_viscosity / np.sqrt(shear_stress*density)
    dimensionless_y = height_above_bed*np.sqrt(shear_stress*density) / dynamic_viscosity    
    dimensionless_u = mean_flow_vel*np.sqrt(density / shear_stress)
    right_side = dimensionless_u + 0.1108*(np.exp(0.4*dimensionless_u)-1-0.4*dimensionless_u - (0.4*dimensionless_u)**2/2 - (0.4*dimensionless_u)**3 / 6)
    
    #find the best shear stress by comparing the two sides and finding min difference
    idx = np.argmin(abs(dimensionless_y - right_side))
    shear_stress_best = shear_stress[idx]
    diff = abs(dimensionless_y[idx] - right_side[idx])
    
    while diff > 10:
        print("couldn't find a good match! Extending trial tau values")
        shear_stress = np.logspace(-10,0,1e6)
        
        height_above_bed = height_above_bed_dimless*dynamic_viscosity / np.sqrt(shear_stress*density)
        dimensionless_y = height_above_bed*np.sqrt(shear_stress*density) / dynamic_viscosity    
        dimensionless_u = mean_flow_vel*np.sqrt(density / shear_stress)
        right_side = dimensionless_u + 0.1108*(np.exp(0.4*dimensionless_u)-1-0.4*dimensionless_u - (0.4*dimensionless_u)**2/2 - (0.4*dimensionless_u)**3 / 6)
        
        #find the best shear stress by comparing the two sides and finding min difference
        idx = np.argmin(abs(dimensionless_y - right_side))
        shear_stress_best = shear_stress[idx]
        diff = abs(dimensionless_y[idx] - right_side[idx])
        print('New diff: '+str(diff)+r', Final $\tau$ = '+str(shear_stress_best)+' Pa')
        
    shear_vel = mean_flow_vel/dimensionless_u[idx]
    #print(height_above_bed[idx],shear_vel,diff,shear_stress[idx],dimensionless_y[idx])

    return shear_stress_best
           
###################################################################################################
def erosion_rate_generic(slope, channel_width, sediment_supply, discharge, grain_size, tensile_strength,
                            density_fluid,density_grain,grav_acc,rock_resistance,youngs_modulus,
                            viscosity,material,special_number,stress_nondim_threshold):
    """calculates erosion rate following Sklar and Dietrich 2004"""
    
    #constants
    buoyancy = density_grain/density_fluid - 1
    sediment_supply_per_width = sediment_supply / channel_width #kg/s/m
    fall_velocity = settling_velocity(grain_size,density_grain,density_fluid,viscosity,grav_acc,0.8,3.5)
    
    #hydraulic radius
    hydraul_radius = calc_hydraulic_radius(discharge, channel_width, grav_acc, slope, material, special_number)
    flow_depth = channel_width*hydraul_radius / (channel_width - 2*hydraul_radius)
    mean_flow_speed = discharge / (flow_depth*channel_width)
    #print(hydraul_radius)
    
    #shear stress
    shear_stress = density_fluid * grav_acc * hydraul_radius * slope
    stress_nondim = shear_stress / ((density_grain - density_fluid)*grav_acc*grain_size)
    transport_stage = stress_nondim/stress_nondim_threshold
    
    shear_stress_factor = stress_nondim / stress_nondim_threshold - 1
    shear_stress_factor[shear_stress_factor < 0] = 0
    
    #fraction of bed exposed
    #sediment_supply_alluviated_per_width = 5.7 * density_grain * np.sqrt(buoyancy * grav_acc * grain_size**3) * (stress_nondim - stress_nondim_threshold)**(3/2)
    sediment_supply_alluviated_per_width = 5.7 * density_grain * np.sqrt(buoyancy * grav_acc * grain_size**3) * stress_nondim_threshold**(3/2)*shear_stress_factor**(3/2)
    frac_exposed = 1 - sediment_supply_per_width/sediment_supply_alluviated_per_width
 
    if hasattr(frac_exposed, "__len__") == False: #checking if incision rate is an array or scalar
        if frac_exposed < 0:
            frac_exposed = 0
        if frac_exposed > 1:
            frac_exposed = 1
    else:
        frac_exposed[frac_exposed < 0] = 0 
        frac_exposed[frac_exposed > 1] = 1
    
    #velocity of impacts 
    flow_shear_velocity = np.sqrt(shear_stress / density_fluid) 
    velocity_factor = 1 - (flow_shear_velocity / fall_velocity)**2

    if hasattr(velocity_factor, "__len__") == False: #checking if incision rate is an array or scalar
        if velocity_factor < 0:
            velocity_factor = 0
    else:
        velocity_factor[velocity_factor < 0] = 0 
    
    #overall incision rate!!!
    incision_rate = 0.08*buoyancy*grav_acc*youngs_modulus / (rock_resistance*tensile_strength**2)*sediment_supply_per_width*shear_stress_factor**(-0.52)*frac_exposed*velocity_factor**(3/2) #Eqn 24a
    
    #components
    prefactor = np.pi*density_grain*buoyancy*grav_acc*grain_size**4*youngs_modulus / (6/0.64*rock_resistance*tensile_strength**2)
    volume_eroded = prefactor*shear_stress_factor**0.36*velocity_factor #Eqn 21
    prefactor2 = 3*sediment_supply_per_width / (4*np.pi*density_grain*grain_size**4)
    impact_rate = prefactor2 * shear_stress_factor**-0.88*np.sqrt(velocity_factor) #Eqn 22
		
    #non-dimensional form of eqn
    k3 = 0.46*(buoyancy*stress_nondim_threshold)**(3/2)/rock_resistance
    stress_non_dim = transport_stage*stress_nondim_threshold
    non_dim_sed = sediment_supply_per_width/sediment_supply_alluviated_per_width
    non_dim_sed[(non_dim_sed)>1] = 1
    incision_rate_nondim = k3*non_dim_sed*(1-non_dim_sed)*shear_stress_factor*(velocity_factor)**(3/2)
    incision_rate_2 = incision_rate_nondim*density_grain*youngs_modulus*(grav_acc*grain_size)**(3/2)/(tensile_strength**2) #Eqn 28

    """
    #checking if shear stress is above the critical value
    if hasattr(incision_rate, "__len__") == False: #checking if incision rate is an array or scalar
        if stress_nondim < stress_nondim_threshold:
            incision_rate = 0
            incision_rate_2 = 0
    else:
        if np.where((stress_nondim - stress_nondim_threshold) < 0): #if there are any places where the stress was below the threshold...
            incision_rate[(stress_nondim - stress_nondim_threshold) < 0] = 0  #...set the erosion rate there to zero
            incision_rate_2[(stress_nondim - stress_nondim_threshold) < 0] = 0  #...set the erosion rate there to zero
	"""
    return incision_rate, incision_rate_2, mean_flow_speed#, hydraul_radius, frac_exposed, sediment_supply_alluviated_per_width
###################################################################################################

###################################################################################################
def erosion_rate_better(slope, sediment_supply_per_width, mean_flow_speed, grain_size, tensile_strength,
                            density_fluid,density_grain,grav_acc,rock_resistance,youngs_modulus,
                            viscosity,material,special_number,stress_nondim_threshold,hydraul_calc):
    """calculates erosion rate following Sklar and Dietrich 2004. Improved starting on 1/17 to (1) use mean flow speed instead of discharge as an input and (2)."""
        
    #constants
    buoyancy = density_grain/density_fluid - 1
    
    #hydraulic radius
    if hydraul_calc == 'yes':
        hydraul_radius = calc_hydraulic_radius_with_meanflowspeed(mean_flow_speed, grav_acc, slope, material, special_number)
    else:
        hydraul_radius = np.asarray([hydraul_calc])
       
    #shear stress
    shear_stress_SD04 = density_fluid * grav_acc * hydraul_radius * slope
    shear_stress = np.asarray([spalding_shear_stress(mean_flow_speed,density_fluid,viscosity,2000)])
    stress_nondim = shear_stress / ((density_grain - density_fluid)*grav_acc*grain_size)
    transport_stage = stress_nondim/stress_nondim_threshold
    
    shear_stress_factor = stress_nondim / stress_nondim_threshold - 1
    shear_stress_factor[shear_stress_factor < 0] = 0   
    
    #fraction of bed exposed
    sediment_supply_alluviated_per_width = 5.7 * density_grain * np.sqrt(buoyancy * grav_acc * grain_size**3) * stress_nondim_threshold**(3/2)*shear_stress_factor**(3/2)

    frac_exposed = 1 - sediment_supply_per_width/sediment_supply_alluviated_per_width
 
    if hasattr(frac_exposed, "__len__") == False: #checking if incision rate is an array or scalar
        if frac_exposed < 0:
            frac_exposed = 0
        if frac_exposed > 1:
            frac_exposed = 1
    else:
        frac_exposed[frac_exposed < 0] = 0 
        frac_exposed[frac_exposed > 1] = 1
    
    #velocity of impacts 
    flow_shear_velocity = np.sqrt(shear_stress / density_fluid)
    fall_velocity = settling_velocity(grain_size,density_grain,density_fluid,viscosity,grav_acc,0.8,3.5) 
    velocity_factor = 1 - (flow_shear_velocity / fall_velocity)**2

    if hasattr(velocity_factor, "__len__") == False: #checking if incision rate is an array or scalar
        if velocity_factor < 0:
            velocity_factor = 0
    else:
        velocity_factor[velocity_factor < 0] = 0 
    
    #flag to understand why the erosion rate is zero
    why_zero = []
    for i in range(len(shear_stress_factor)):
        why_zero.append(i)
        if shear_stress_factor[i] == 0:
            why_zero.append('shear stress')
        if frac_exposed[i] == 0:
            why_zero.append('frac exposed')
        if velocity_factor[i] == 0:
            why_zero.append('velocity')
    
    #print('Shear Stress: '+str(shear_stress)+' Pa, Nondim: '+str(stress_nondim))
    print(why_zero)
    
    #overall incision rate!!!
    incision_rate = 0.08*buoyancy*grav_acc*youngs_modulus / (rock_resistance*tensile_strength**2)*sediment_supply_per_width*shear_stress_factor**(-0.52)*frac_exposed*velocity_factor**(3/2) #Eqn 24a
    
    
    #components
    prefactor = np.pi*density_grain*buoyancy*grav_acc*grain_size**4*youngs_modulus / (6/0.64*rock_resistance*tensile_strength**2)
    volume_eroded = prefactor*shear_stress_factor**0.36*velocity_factor #Eqn 21
    prefactor2 = 3*sediment_supply_per_width / (4*np.pi*density_grain*grain_size**4)
    impact_rate = prefactor2 * shear_stress_factor**-0.88*np.sqrt(velocity_factor) #Eqn 22
		
    #non-dimensional form of eqn
    k3 = 0.46*(buoyancy*stress_nondim_threshold)**(3/2)/rock_resistance
    stress_non_dim = transport_stage*stress_nondim_threshold
    non_dim_sed = sediment_supply_per_width/sediment_supply_alluviated_per_width
    non_dim_sed[(non_dim_sed)>1] = 1
    incision_rate_nondim = k3*non_dim_sed*(1-non_dim_sed)*shear_stress_factor*(velocity_factor)**(3/2)
    incision_rate_2 = incision_rate_nondim*density_grain*youngs_modulus*(grav_acc*grain_size)**(3/2)/(tensile_strength**2) #Eqn 28
    """
    #checking if shear stress is above the critical value
    if hasattr(incision_rate, "__len__") == False: #checking if incision rate is an array or scalar
        if stress_nondim < stress_nondim_threshold:
            incision_rate = 0
            incision_rate_2 = 0
    else:
        if np.where((stress_nondim - stress_nondim_threshold) < 0): #if there are any places where the stress was below the threshold...
            incision_rate[(stress_nondim - stress_nondim_threshold) < 0] = 0  #...set the erosion rate there to zero
            incision_rate_2[(stress_nondim - stress_nondim_threshold) < 0] = 0  #...set the erosion rate there to zero
	"""
    return incision_rate, incision_rate_2,why_zero, hydraul_radius#, frac_exposed, sediment_supply_alluviated_per_width



###################################################################################################

###################################################################################################
def erosion_rate_final(sediment_supply_per_width, mean_flow_speed, grain_size, tensile_strength,
                            density_fluid,density_grain,grav_acc,rock_resistance,youngs_modulus,
                            viscosity,stress_nondim_threshold):
    """calculates erosion rate following Sklar and Dietrich 2004. THIS VERSION IS USED FOR THE PAPER"""
        
    #constants
    buoyancy = density_grain/density_fluid - 1
       
    #shear stress
    shear_stress = np.asarray([spalding_shear_stress(mean_flow_speed,density_fluid,viscosity,2000)])
    stress_nondim = shear_stress / ((density_grain - density_fluid)*grav_acc*grain_size)
    transport_stage = stress_nondim/stress_nondim_threshold
    
    shear_stress_factor = stress_nondim / stress_nondim_threshold - 1
    shear_stress_factor[shear_stress_factor < 0] = 0   
    
    #fraction of bed exposed
    sediment_supply_alluviated_per_width = 5.7 * density_grain * np.sqrt(buoyancy * grav_acc * grain_size**3) * stress_nondim_threshold**(3/2)*shear_stress_factor**(3/2)

    frac_exposed = 1 - sediment_supply_per_width/sediment_supply_alluviated_per_width
 
    if hasattr(frac_exposed, "__len__") == False: #checking if incision rate is an array or scalar
        if frac_exposed < 0:
            frac_exposed = 0
        if frac_exposed > 1:
            frac_exposed = 1
    else:
        frac_exposed[frac_exposed < 0] = 0 
        frac_exposed[frac_exposed > 1] = 1
    
    #velocity of impacts 
    flow_shear_velocity = np.sqrt(shear_stress / density_fluid)
    fall_velocity = settling_velocity(grain_size,density_grain,density_fluid,viscosity,grav_acc,0.8,3.5) 
    velocity_factor = 1 - (flow_shear_velocity / fall_velocity)**2

    if hasattr(velocity_factor, "__len__") == False: #checking if incision rate is an array or scalar
        if velocity_factor < 0:
            velocity_factor = 0
    else:
        velocity_factor[velocity_factor < 0] = 0 
    
    #flag to understand why the erosion rate is zero
    why_zero = []
    for i in range(len(shear_stress_factor)):
        why_zero.append(i)
        if shear_stress_factor[i] == 0:
            why_zero.append('shear stress')
        if frac_exposed[i] == 0:
            why_zero.append('frac exposed')
        if velocity_factor[i] == 0:
            why_zero.append('velocity')
    
    #print('Shear Stress: '+str(shear_stress)+' Pa, Nondim: '+str(stress_nondim))
    #print(why_zero)
    
    #overall incision rate!!!
    incision_rate = 0.08*buoyancy*grav_acc*youngs_modulus / (rock_resistance*tensile_strength**2)*sediment_supply_per_width*shear_stress_factor**(-0.52)*frac_exposed*velocity_factor**(3/2) #Eqn 24a
    
    return incision_rate, why_zero, flow_shear_velocity


###################################################################################################
"""
density_grain = [2300,2650,2900] #density of quartz in kg/m^3
density_fluid = 1000 #kg/m^3
grav_acc = 1.41 #gravity
diameter = 0.01 #m, following SD04
youngs_modulus = 5e10 #1.1e-6Pa, SD04
rock_resistance = 1e6 #unitless, SD04
tensile_strength = 7e6 #Pa, SD04
#discharge = np.asarray([39.1])
mean_flow_speed = [1]#np.logspace(-5,0,100)#np.asarray([2.37])
roughness = diameter#0.035
channel_width = 1 #m
slope = [0.001,0.01,0.1]
kinematic_viscosity = 1.14e-6
sediment_supply_per_width = np.logspace(-10,0,1000)#[0.01,0.1,1,10] #kg/width/time

#incision_rate, incision_rate_from_nondim,mean_flow_speed = erosion_rate_generic(slope, channel_width, sediment_supply, discharge, diameter, tensile_strength,density_fluid,density_grain,grav_acc,rock_resistance,youngs_modulus,kinematic_viscosity,'geoff',roughness,0.03)

#print(incision_rate,incision_rate_from_nondim,mean_flow_speed)

plt.figure()

col = ['thistle','mediumpurple','indigo']
ls = [':','-','--']



incision_rate_from_nondim = np.zeros_like(sediment_supply_per_width)
for i_density in range(len(density_grain)):
    for i_slope in range(len(slope)):
        for i in range(len(sediment_supply_per_width)):
            incision_rate, incision_rate_from_nondim[i], x, y = erosion_rate_better(slope[i_slope], channel_width, sediment_supply_per_width[i], mean_flow_speed, diameter,   tensile_strength, density_fluid, density_grain[i_density], grav_acc, rock_resistance, youngs_modulus, kinematic_viscosity, 'choose for me', 6.8*roughness, 0.03)

            #print(incision_rate,incision_rate_from_nondim,mean_flow_speed,x)
            #print(incision_rate)
            #print(x)

        plt.plot(sediment_supply_per_width,incision_rate_from_nondim*1000*365.25*24*3600,label = 'Slope:'+str(slope[i_slope])+' m/m, Density = '+str(density_grain[i_density]),color = col[i_slope],linewidth = 3,linestyle = ls[i_density])
 
print(incision_rate_from_nondim) 
    
plt.xlabel('Sediment Supply [kg/m/s]',fontsize = 14)
plt.ylabel('Erosion Rate [km/My]',fontsize = 14)
plt.tick_params(labelsize = 12)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.xlim([1e-10,1])
plt.ylim([1e-10,1])

plt.tight_layout()
plt.show()
"""

"""
mean_flow_speed = np.logspace(-5,0,100)
channel_width = 10
grav_acc = 0.133
slope = 0.005
material = 'gravel'
special_number = 0.06

rh = calc_hydraulic_radius_with_meanflowspeed(mean_flow_speed, channel_width, grav_acc, slope, material, special_number)

plt.plot(mean_flow_speed,rh)
plt.show()
"""
            

