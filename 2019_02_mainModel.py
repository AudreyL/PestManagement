#!/usr/bin/env python3

#################################################################################
# Simulation scenarios
#################################################################################
# Audrey Lustig
# January 2019
#  Spread model for predicting the distribution and abundance of mammalian pests across the landscape, the ways in which animals move from their natal sites, and the effects of control intervention. In particular, the model can help managers to asses the chances of success of a management action.
# Parameters are set to average data for possums
# Half of the landscape (treatment area) is under homogeneous control, in the other half the population is left undisturbed and at carrying capacity
# 20 individuals are randomly introduced in the treatment area

# Population dynamics followed for 10 years -  100 replications



###### Import pyhton librairies
from __future__ import division
import random
import numpy as np
import scipy.stats
from scipy import stats
import numba
import multiprocessing
import itertools
import sys
import glob, os
import gzip
import shutil

# Load function
# Allow to read and rescale ascii grid
#sys.path.insert(0, '/home/audrey/Documents/github') # set path to folder
import habitat_suitability 


#### General variables (random walk)
# Valid moves for each compass direction NESW / neighboor
moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
# Valid coutermoves for each compass direction NESW / neighboor
countermov=[2, 3, 0, 1] 


# Beta-PERT distribution to approximate birth events
def pertDistributionRVS(xmin, xmode,  xmax, lambdas = 4, size=1):
	'''
	Random variates sampling in a Beta pert distribution
	i.e. special case of a beta distribution that takes three parameters: a minimum, maximum, and most likely (mode). 
	'''
	if( xmin > xmax or xmode > xmax or xmode < xmin ):
		raise ValueError("Invalid reproduction parameters, rMmin < rMode < rMax")
	rangex = 1.0*(xmax - xmin)
	if (rangex == 0):
		return [xmin] * size
	mu = (xmin + xmax + lambdas * xmode ) / ( 1.0*lambdas + 2.0 ) 
	# special case if mu == mode
	if( mu == xmode ):
		v = ( lambdas / 2.0 ) + 1
	else:
		v = (( mu - xmin ) * ( 2.0 * xmode - xmin - xmax )) / (1.0*(( xmode - mu ) * ( xmax - xmin )))
	w = ( v * ( xmax - mu )) / (1.0*( mu - xmin ))
	return scipy.stats.beta.rvs(v, w, size=size) * rangex + xmin

# Weighted probability/ movement choice function
def weighted_choice(weights):
	'''
	Given a list of numerical weights, return an index with probability weights[i]/sum(weights).
	Adapted from http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
	'''
	if all([w==0 for w in weights]):
		return random.randint(0, len(weights)-1)
	else:
		rnd=random.random() * sum(weights)
		for i, w in enumerate(weights):
			rnd -= w
			if rnd < 0:
				return i

# Load library Numba, high performance python compiler
@numba.jit
def choice_move(x, y, memory, e, moves):
	'''Dispersal was modeled has a stochastic process. Dispersing animals interrogate the landscape and move cell-by-cell through the cell in the landscape. The choice of the direction is bias toward habitat of higher quality. Juveniles were not allowed to return in the same direction from which they had come unless there was no suitable dispersal habitat ahead of then. A map of carrying capacity is given in input, area not suitable for home-range but that can be dispersed (like river) are given a dispersal probabilit of 0.2 (min prob). Are that are not suitable for dispersal or home-range were already encoded by the value -99. Area outside the extent of the study are encoded -101. '''
	adjacent_positions=[]	# Extract coordinates of adjacent cells
	for move in moves:
		adjacent_positions.append([x, y] + np.array(move)) # add coordinates of adjacent list to adjacent_position
	IndexToRemove = []
	adjacent_qualities=np.zeros(len(adjacent_positions)) # to extract carrying capacity K of adjacent cells 
	for nn in range(len(adjacent_positions)):  # loop over adjacent cells
		(newx, newy) = adjacent_positions[nn]
		if newx < 0 or newx >= e.Esize[0] or newy < 0 or newy >= e.Esize[1] : # In the case I go outside the study area
			IndexToRemove.append(nn) # remove index
			adjacent_qualities[nn]=-101 # set quality to -101
		else:
			adjacent_qualities[nn] = e.resized_habitat[newx, newy]  # Otherwise extract carrying capacity K of adjacent cells 
	adjacent_qualities[(adjacent_qualities==0)]=0.2 # update the quality of area not suitable for home-range but Ok for dispersal to the min carrying capacity (area not suitable for dispersal have been set to a carrying capacity -99)
	if sum(adjacent_qualities[adjacent_qualities>0]) !=0 : # update the quality of the cell oustide the extent of the area to the mean habitat quality (boundary conditions)
		adjacent_qualities[(adjacent_qualities==-101)]=sum(adjacent_qualities[adjacent_qualities>0])/(1.0*len(adjacent_qualities[adjacent_qualities>0])) 
	else:
		adjacent_qualities[(adjacent_qualities==-101)]=0
	adjacent_qualities[adjacent_qualities<0]=0 # update the quality of area not suitable for dispersal (input -99) to 0 as sea is not suitable for dispersal (i.e. choose the cell with prob 0!)
	if memory != None: 
		adjacent_qualities[countermov[memory]]=0 # Idenitified previsous location and set probability of dispersal to 0
	if sum(adjacent_qualities) != 0 : # if there is a possibility to move forward, I move forward
		adjacent_qualities=adjacent_qualities/(1.0*sum(adjacent_qualities))
		move_index = weighted_choice(adjacent_qualities) # choose where to move (previous location has a probability zero)
	else: # if individuals are facing an obstacle, I move backward! (for ex sea or estuary)
		move_index = countermov[memory] # otherwise I return
	return list((move_index,adjacent_positions[move_index]))



class offsprings(object):
	'''
	Animal with position in environment
	x,y = position /  current coordinates in the landscape
	xx,yy = record initial position to determine whether the juveniles is born wihtin the tretament area or not
	state = dispersing, trapped or outside (the tretament area)
	position_history = follow mvt of each individuals cell-by-cell 
	memory = meomory of previous location as juvenile are not allowed to return in the same direction from which they had come

	'''
	def __init__(self, xx, yy, x,y,e):
		self.initx = xx
		self.inity = yy
		self.x = x # coordinates x
		self.y = y # coordinates y
		self.state='dispersing'
		self.position_history=[[self.x,self.y]] # follow trajectory in terms of position in the landscape
		self.memory = None # initiate memory to none
	def trap_juvenile(self,e): # can have params of the bimodal distribution as a parameter 
		''' trapping function for juveniles'''
		rho = e.trap_density[self.x,self.y] # trap density at location x,y
		if rho >0: # if trap density > 0
			Pavoid = np.exp(- e.A * rho * e.g1) # caclulcate the prbability of avoiding a trap within 1 night as function of trap density
			Ptrap = float(1 - Pavoid) 
			Puni = np.random.uniform(0,1)
			if Ptrap >= Puni: # if the juvenile is trapped
				self.state = 'trapped'
				e.offsrpingDistribution[self.x,self.y]-=1 # remove the individual from offspring distribution
				e.juveniles_trapped_loc[self.x,self.y]+=1 # add count to juvenile trapped map
				e.juveniles_trapped_count +=1 # add count to juvenile trapped
	def move(self,e,simulationTime):
		if simulationTime % e.timestep_perYear in e.trapping_session: # only trap during active trapping session
			self.trap_juvenile(e) # trap juvenile
		if self.state == 'dispersing': # if I have not been trapped I can disperse
			Puni = np.random.uniform(0,1)
			if e.resized_habitat[self.x,self.y] == 0.0: # in the case the inidividual is in a dispering habitat (can't settle)
				Pstay = 0 # The individual keep moving
			else:
				Pdensity_dependence =1.0-(e.density_dependence*(e.distribution[self.x,self.y])/(1.0*e.resized_habitat[self.x,self.y])) # Calculate habitat/density dependence (see Lustig et al.  2019)
				if e.habitat_dependence == 0:
					Phabitat_quality = 1
				else:
					Phabitat_quality = 1-(1.0/(1.0+np.exp((e.habitat_dependence*(e.resized_habitat[self.x,self.y])/np.mean(e.resized_habitat[e.resized_habitat>0]))-1.0)))
				Pstay = 1 - pow((1- Pdensity_dependence*Phabitat_quality),(e.resolution/(1.0*e.max_resolution))) # Needs to be at power of R/Rmax to maintain proba over different spatial scale.
			if Pstay >= Puni and (e.distribution[self.x,self.y]+1)<=e.resized_habitat[self.x,self.y]: # if Pstay and local density below local carrying capacity
			#if 0.3 >= Puni: # to test that 30% of the individual stay and other disperse to the closest cell
				self.state = 'settled' # settle
				e.youngAdultdistribution[self.x,self.y]+=1 # Add individual to young adult population (won't reproduce in current year)
				e.pattern_dispersion[self.x,self.y]+=1 # add one individual to dispersal pattern 
				e.distribution[self.x,self.y]+=1 # add one individual to total distribution
				e.offsrpingDistribution[self.x,self.y]-=1 # remove individual from offsrping distribution
				if e.my_study_area[self.initx,self.inity]>0 and e.my_study_area[self.x,self.y]>0: # if I am born in the treatment area and settle in the treatment area
					e.nbsettling +=1 # count animal setteling in treatment area
				if e.my_study_area[self.initx,self.inity]>0 and e.my_study_area[self.x,self.y]==0: # if I am born in the treatment area and settle outside the treatment area 
					e.emmigration +=1 # count animal emmigrating from the treatment area
				if e.my_study_area[self.initx,self.inity]==0 and e.my_study_area[self.x,self.y]>0: # if I am born outside the treatment area and settle in the treatment area
					e.immigration +=1 # count animal imigrating to the treatment area
			else: # if the inidvidual did not settle
				alea=choice_move(self.x, self.y, self.memory, e, moves) # choose direction of mvt
				(xx,yy) = alea[1].tolist() # transform to coordinate
				self.memory = alea[0] # update mvt memory
				if xx >= e.Esize[0] or xx<0 or yy >=e.Esize[1] or yy<0 or e.resized_habitat[xx,yy]==-101: # identify coordinate outside the extent of the study area
					self.state = 'outsider' # outsider or indivdiual 
					e.offsrpingDistribution[self.x,self.y]-=1 # remove individual from offsrping population
				else: # if I can still disperse, I keep dispersing
					e.offsrpingDistribution[self.x,self.y]-=1 # remove offsrping from last position	
					(self.x,self.y)=(xx,yy) # update coordinates of the offspring
					e.offsrpingDistribution[self.x,self.y]+=1 # add offsrping at the new location
					e.pattern_dispersion[self.x,self.y]+=1 # add offspring to the dispersion patterns
					self.position_history.append([self.x,self.y]) # record the position of the offspring


class envir(object):
	'''
	Animal with position in environment
	parameters: raster of carrying capacity of resolution 100 m, raster of the current treatment area of resolution 100 m, nb replication, spatial resolution, max spatial resolution, name of studied pest, min repductivity, mean reproductivity, max reproductivity, life span (year), gammma0 (adult trapability), sigma (adult trapability) , gamma1 (juvenile trapability), maximum dispersal distance (m), habitat dependence relationship (default 1), density dependence relationship (default 1),  size of initial population (number of indivdiuals),  raster of trap location (density per cell, 100m resolution) or number (homogeneous distribution), number of trapping night per months, trapping freqency per year, time of simulation (years), name of output folder'''
	def __init__(self, my_habitat_K,my_study_area, nbrep, resolution, max_resolution, predator, Fmin, Fmean, Fmax, life_span, gamma0, sigma, gamma1, max_distance, habitat_dependence, density_dependence, initPopSize,  my_trap_loc, nb_nights, trapping_frequency, simulationTime, output_distribution):
		##### Map of carrying capacity
		self.output_distribution=output_distribution # output folder
		self.nbrep = nbrep # replication number
		self.resolution=resolution # choosen resolution
		# resize entry maps
		self.max_resolution = max_resolution # maximum resolution possible (determined by the velocity of the individuals)
		self.resized_habitat=habitat_suitability.bin_ndarray(my_habitat_K,100, resolution, operation='sum')/2.0 # resized habitat based on the choosen resolution - the reference map should be of resolution 500 for the top predator of NZ (resolution 100 m)
		self.my_study_area = habitat_suitability.bin_ndarray(my_study_area,100, resolution, operation='sum').astype(int) # raster of study area: 0 oustide, 1 inside (resolution 100 m)
		self.Esize = self.resized_habitat.shape # Shape/dimension of the new map of carrying capacity
		if isinstance(my_trap_loc, np.ndarray): # if the user give a raster file of trap density (resolution 100 m)
			my_trap_loc_resized = habitat_suitability.bin_ndarray(my_trap_loc, 100, resolution, operation='sum') # resize trap density raster: to make sure both local carrying capacity and trapping density are of the same resolution
		else: # create a raster of trap density with homogeneous distribution over the landscape
			my_trap_loc_resized = np.zeros(self.resized_habitat.shape) + my_trap_loc*(self.resolution/100.0)**2 # set homogeneous density of trap in the landscape
			my_trap_loc_resized[self.my_study_area==0] = 0 # but only for the treatment area, outside the boundaries set to 0!
		self.trap_density = my_trap_loc_resized/(float(self.resolution)**2) # transform number of traps to trap density
		self.initPopSize  = initPopSize # number of adults introduced at the beginning of the simulation
		if self.initPopSize >  np.sum(self.resized_habitat[self.resized_habitat>0]):	# if number above carrying capacity, raise error
			raise Exception("The initial population size is above the total habitat carrying capacity.")
			sys.exit(1)
		##### Simulation time
		self.timestep_perYear = 12 # number of time step per year (if 12 monthly time step!)
		self.simulationTime=simulationTime * self.timestep_perYear # simulation time in month rather than in year
		##### Population variables
		self.distribution = np.zeros(self.Esize, int) # Distribution of adults
		self.pattern_dispersion= np.zeros(self.Esize, int) #Pattern of dispersion
		self.youngAdultdistribution  = np.zeros(self.Esize, int) # Distribution of young adults
		self.offsrpingDistribution = np.zeros(self.Esize, int) # Distribution of offsprings during dispersal (initialised at zero at the begining of each year)
		self.adultReproduction = np.zeros(self.Esize, int) # Number of adults reproducing at the begining of the year
		self.juveniles_trapped_loc= np.zeros(self.Esize, int)	# Hot map of juveniles trapped in the landscape
		self.adults_trapped_loc= np.zeros(self.Esize, int)	# Hot map of adults trapped in the landscape
		self.Fmin=Fmin # minimum reproductive rate
		self.Fmean=Fmean # mean reproductive rate
		self.Fmax=Fmax # max reproductive rate
		self.life_span = life_span # average life span of the individual
		self.max_distance = np.floor(max_distance/(1.0*self.resolution)).astype(int) # maximal dispersal distance of offsrpings
		self.habitat_dependence = habitat_dependence # modulate the influence of landscape heterogenity on mvt of offspring
		self.density_dependence = density_dependence # modulate the density dependent function
		##### Trapping parameters
		self.g0 = gamma0  # the probability of capture of an individual by a trap placed at the animal’s home-range centre
		self.g1 = gamma1  # the probability of capture of an offspring during dispersal
		self.sig = sigma # he spatial decay parameter for a half normal home-range kernel
		self.nb_nights = nb_nights # the number of nights trapping
		self.trapping_session = np.linspace(0,12,trapping_frequency).astype(int) # Month when trapping occur
		##### output parameters, we follow output at the monthly scale, not at a daily scale
		self.nbOffsprings = 0 # number of offspring at the beginning of the simulation
		self.birthPerYear = 0 # brith count 
		self.nbsettling = 0  # number of offspring setteling each month
		self.emmigration = 0 # number of offspring emmigrating from C2C each month
		self.immigration = 0 # number of offspring imigrating in C2C each month
		self.juveniles_trapped_count = 0 # number of juveniles trapped per trapping session
		self.offspringPop = [] # Population of offspring
		self.nbAdults_reproducing = np.zeros(self.simulationTime, int) # number of individuals reproducing each month(output variable)
		self.densityPerMonth = np.zeros(self.simulationTime, float) # density per month
		self.nbBirthPerMonth = np.zeros(self.simulationTime, int) # number of birth each month  (output variable)
		self.settlingRate = np.zeros(self.simulationTime, int) # number of offspring setteling each month (output variable)
		self.dipsersalMortality = np.zeros(self.simulationTime, int) # offspirng mortality reported each month (output variable)
		self.emigrationRate = np.zeros(self.simulationTime, int) # number of offspring imigrating in C2C each month (output variable)
		self.imigrationRate = np.zeros(self.simulationTime, int) # number of offspring imigrating in C2C each month(output variable)
		self.nbAdults_death = np.zeros(self.simulationTime, int) # adult mortality reported each month (output variable)
		self.nbAdults_trapped = np.zeros(self.simulationTime, int) # number of adults trapped each month  (output variable)
		self.nbjuveniles_trapped = np.zeros(self.simulationTime, int) # number of juveniles trapped each month (output variable)
		self.dispersal_distance = [0.0,0.0,0.0] # vector to store dispersal distance (time step, distance, euclidean_distance)
		self.fullpop_trajectory = [] # record trajectory of every individuals that settle during dispersal to claculate distance and euclidean distance
		# Reproduction function
		if predator == 'possum':
			gammma0 = 2 
			theta0 = 1.2 
			lambda0 = 0.55
			gammma1 = 8
			theta1 =  0.4
			lambda1 = 0.04
			x1 = np.linspace(0,13,10000)
			Gauss1 = lambda0*scipy.stats.norm.pdf(x1,gammma0,theta0) 
			Gauss2 = lambda1*scipy.stats.norm.pdf(x1,gammma1,theta1) 
			bdistribution=[sum(Gauss1[(tt-1 <= x1) & (x1 < tt)])+sum(Gauss2[(tt-1 <= x1) & (x1 < tt)])  for tt in np.linspace(1,12,num=self.timestep_perYear)]+ np.random.uniform(0.0, 0.05,self.timestep_perYear)
			self.birth_distribution=(bdistribution)/sum(bdistribution)
			self.A = 2 * 0.0347 * (14*24*60*60)/(self.max_distance)  # A = W V dt with dt = tmax * R / dmax
	################# Initialising the population
	def populate_predators(self):
		
		# Initiate the area outside C2C to carrying capacity
		self.distribution[self.my_study_area==0] = self.resized_habitat[self.my_study_area==0]*30/100.0 # Outside th treatement area we assume the population at 30% of its carrying capacity
		self.distribution[self.distribution<0] = 0 # but set to 0 outside not suitable for settling home-range
		self.pattern_dispersion[self.my_study_area==0] = self.resized_habitat[self.my_study_area==0] # initiate dispersal patterns
		self.pattern_dispersion[self.pattern_dispersion<0]=0
		index_inC2C=np.nonzero(self.my_study_area)
		# for indivdiuals inside the treatment area, introduce 'initPopSize' individuals at random location
		for i in range(self.initPopSize):
			[xinit,yinit] = np.array([random.choice(index_inC2C[0]),random.choice(index_inC2C[1])]) # initiate population at random
			while (self.distribution[xinit,yinit]+1) > self.resized_habitat[xinit,yinit]: # check introduced number of adults per cell is under the local carrying capacity  
				[xinit,yinit] = np.array([random.choice(index_inC2C[0]),random.choice(index_inC2C[1])])
			self.distribution[xinit,yinit]+=1 # add individual to distribution pattern
			self.pattern_dispersion[xinit,yinit]+=1 # add inidvidual to dispersal pattern
	################# Population dynamic
	# Create the bimodal distribution of birth events
	def reproduction_func(self, n, simulationTime): # reproduction function take the number of inidviduals and time of the year as entry parameters
		nb_babies=pertDistributionRVS(self.Fmin, self.Fmean, self.Fmax, lambdas = 4, size=1) # beta-Pert probability use to determine number of offsprings (Glen et al. 2017)
		return(np.random.binomial(n, nb_babies*self.birth_distribution[simulationTime % self.timestep_perYear])) # return the number of offsprings with a probability changing during month of the year
	# IMplement reproduction function for each raster cell
	def reproduce(self, simulationTime):	
		if simulationTime % self.timestep_perYear == 0: 	# at the begining of the year count the number of adult that can reproduce
			index=np.nonzero(self.distribution) # Extract location where adults are present
			self.adultReproAdult = np.zeros(self.Esize, int) # initiate a temporary distribution of adults reproducing
			self.adultReproAdult[index] = self.distribution[index] # Fill the distribution of adults reproducing
			# Create a temporary distribution of young adult (juvenile that have settle their home-range but can't reproduce) 
			self.youngAdultdistribution = np.zeros(self.Esize, int) # initiate a temporary distribution of young adults 
		birthPerMonth = 0 # Number of birth at the begining of the month set to zero
		self.nbAdults_reproducing[simulationTime] = np.sum(self.adultReproAdult[self.my_study_area>0]) # set the number of adults reproducing at each time step 
		self.nbsettling = 0 # number of offspring settling
		self.juveniles_trapped_count=0 # number of  offspring trapped
		newborns = self.reproduction_func(self.adultReproAdult, simulationTime) # Number of offspring per raster cell 
		newpop = np.nonzero(newborns) # Set offsping locations
		if len(newpop[0]) > 0:
			for index in range(len(newpop[0])): # for each raster cell with offsprings
				xinit=newpop[0][index] # save coordinate x
				yinit= newpop[1][index] # save coordinate y
				for nboffspring in range(newborns[xinit,yinit]): # for each juvenile (in each raster cell)
					self.offspringPop.append(offsprings(xinit,yinit, xinit,yinit,self)) # add one offspring to the population of offsprings
					self.offsrpingDistribution[xinit,yinit]+=1 # add one offspring to the distribution of offsprings
					self.nbOffsprings +=1 # add one offspring to the offspring count
					if self.my_study_area[xinit,yinit]>0: # if offspring is born in the treatement area
						birthPerMonth +=1 # add one to the number of birth per year in treatment area
		self.nbBirthPerMonth[simulationTime] = birthPerMonth # save the number of birth per year
	def mortality_func(self, n):
		monthly_mortality=float(1.0/(self.life_span*self.timestep_perYear+1.0)) # monthly mortality (constant)
		return(np.random.binomial(n, monthly_mortality)) # determine the number of death per month in every cell
	def mortality(self, simulationTime):	
		death = self.mortality_func(self.adultReproAdult) # determine the number of death per month in every cell. We consider only adults here.
		index=np.nonzero(death)
		self.adultReproAdult[index] -= death[index] # remove the individuals from the reproducing adults
		self.distribution[index] -= death[index]	# remove the individuals from adult distribution
		self.nbAdults_death[simulationTime] = np.sum(death[self.my_study_area>0]) # count number of natural death (adults)
	################# Dispersal
	def move(self,simulationTime):
		self.emmigration = 0 # emmigration is set a zero (number of individuals going outside the treatment area)
		self.immigration = 0 # emmigration is set a zero (number of individuals coming inside the treatment area)
		for dispTime in range(0,self.max_distance):
			outsider_to_remove = [] # individual going oustide the simulation arena
			adults_settled = [] # individual setteling and being recruited to the adult pop
			juveniles_trapped = [] # individual beeing trapped
			for i in range(self.nbOffsprings):
				self.offspringPop[i].move(self,simulationTime) # move
				if self.offspringPop[i].state == 'outsider': # save individual going outside the simulation arena
					outsider_to_remove.append(i)	
				elif self.offspringPop[i].state == 'settled': # save individual setteling
					adults_settled.append(i)
				elif self.offspringPop[i].state == 'trapped': # save individual trapped
					juveniles_trapped.append(i)
			outsider_to_remove=np.array(outsider_to_remove)
			adults_settled=np.array(adults_settled)
			juveniles_trapped = np.array(juveniles_trapped)
			if len(outsider_to_remove)>=1: # remove juveniles going outside the simulation arena from 
				removed_count=0
				for j in outsider_to_remove: 
					if self.my_study_area[self.offspringPop[j-removed_count].initx,self.offspringPop[j-removed_count].inity]>0: # if individual was born inside the treatment area
						self.emmigration +=1 # count one extra emmigrant
					del self.offspringPop[j-removed_count] # remove inidvidual from list of offspring
					if len(adults_settled)>=1:
						adults_settled[adults_settled>=j-removed_count]-=1 # update index of offspring (young adult) settled
					if len(juveniles_trapped)>=1:
						juveniles_trapped[juveniles_trapped>=j-removed_count]-=1 # update index of offspring (young adult) settled
					self.nbOffsprings -=1 # remove individuals from the count of offsprings
					removed_count+=1 # count one individual removed
			if len(juveniles_trapped)>=1: # remove juveniles trapped
				removed_count=0
				for m in juveniles_trapped:
					del self.offspringPop[m-removed_count] # remove individual from juvenile list
					if len(adults_settled)>=1:
						adults_settled[adults_settled>=m-removed_count]-=1 # update index of offspring (young adult) settled
					self.nbOffsprings -=1 # remove individuals from the count of offsprings
					removed_count+=1 # count one individual removed
			if len(adults_settled)>=1:
				removed_count=0
				for k in adults_settled:
					# only record trajectory for offspring in the C2C
					if self.my_study_area[self.offspringPop[k-removed_count].initx,self.offspringPop[k-removed_count].inity]>0 and self.my_study_area[self.offspringPop[k-removed_count].x,self.offspringPop[k-removed_count].y]>0: # if I am born in the treatment area and settle in the treatment area
						self.fullpop_trajectory.append(self.offspringPop[k-removed_count].position_history)
					if self.my_study_area[self.offspringPop[k-removed_count].initx,self.offspringPop[k-removed_count].inity]>0 and self.my_study_area[self.offspringPop[k-removed_count].x,self.offspringPop[k-removed_count].y]==0: # if I am born in the treatment area and settle outside the treatment area
						self.fullpop_trajectory.append(self.offspringPop[k-removed_count].position_history)
					if self.my_study_area[self.offspringPop[k-removed_count].initx,self.offspringPop[k-removed_count].inity]==0 and self.my_study_area[self.offspringPop[k-removed_count].x,self.offspringPop[k-removed_count].y]>0: # if I am born outside the treatment area and settle in the treatment area
						self.fullpop_trajectory.append(self.offspringPop[k-removed_count].position_history)
					del self.offspringPop[k-removed_count] # remove individuals from list of offspring
					self.nbOffsprings -=1 # remove individuals from the count of offsprings
					removed_count +=1 # count one individual removed
		# at the end of the dispersal phase, offpsring that have not found a home-range are assumed to die. 
		juvenile_death =0 
		if self.nbOffsprings>0:
			removed_count=0	
			for i in range(self.nbOffsprings):
				self.nbOffsprings -=1 # remove individuals from the count of offsprings
				if self.my_study_area[self.offspringPop[i-removed_count].initx,self.offspringPop[i-removed_count].inity]>0 and self.my_study_area[self.offspringPop[i-removed_count].x,self.offspringPop[i-removed_count].y]>0:
					juvenile_death +=1 # count one death as result of dispersal
				self.offsrpingDistribution[self.offspringPop[i-removed_count].x,self.offspringPop[i-removed_count].y]-=1  # remove from offspring distribution
				del self.offspringPop[i-removed_count]	# remove individuals from list of offspring
				removed_count+=1 # count one individual removed
		self.emigrationRate[simulationTime] = self.emmigration # update emmigration rate 
		self.imigrationRate[simulationTime] = self.immigration # update imigration rate 
		self.settlingRate[simulationTime] = self.nbsettling # update number of offspring settled
		self.nbjuveniles_trapped[simulationTime] =  self.juveniles_trapped_count # number of offspring trapped
		self.dipsersalMortality[simulationTime] = juvenile_death  # number of offspring that have not found a home-range (natural death)
		# calculate distance and euclidean distance
		distance = [(len(self.fullpop_trajectory[kk])-1) * self.resolution  for kk in range(len(self.fullpop_trajectory))] # calculate average dispersal distance
		euclidean_distance = np.zeros(len(distance),float) # and update euclidean distance
		for mm in range(len(euclidean_distance)):
			if len(self.fullpop_trajectory[mm]) > 1:
				euclidean_distance[mm]=np.sqrt(np.power((self.fullpop_trajectory[mm][len(self.fullpop_trajectory[mm])-1][0]-self.fullpop_trajectory[mm][0][0]),2)+np.power((self.fullpop_trajectory[mm][len(self.fullpop_trajectory[mm])-1][1]-self.fullpop_trajectory[mm][0][1]),2))*self.resolution
		self.dispersal_distance = np.vstack((self.dispersal_distance,np.column_stack((np.zeros(len(distance), int) + simulationTime, distance , euclidean_distance))))
	################# Trapping function
	# define the trapping as a function of trap density in each cell
	def trapping_func(self,n,rho): # the trapping function takes the number of individuals per cell and corresponding trap density as entry parameters
		Pavoid = np.exp(-2 * np.pi * self.g0 * (self.sig**2)* self.nb_nights * rho) # caclulcate the prbability of avoiding a trap within n nights as function of trap density
		trapping_probability = float(1 - Pavoid) 
		return(np.random.binomial(n, trapping_probability))
	# trap adult population
	def trap_adults(self, simulationTime): 
		# Trap for adults
		traps_loc_index = np.nonzero(self.trap_density) # identify ratser cell with traps (rho >0) 
		if np.sum(self.adultReproAdult)>0 and len(traps_loc_index[0])>0:
			N_adults_trapped_perCell = [self.trapping_func(self.adultReproAdult[c,d], self.trap_density[c,d]) for c,d in zip(traps_loc_index[0],traps_loc_index[1])] # Nb adults trapped per cell
			self.distribution[traps_loc_index] = self.distribution[traps_loc_index]- N_adults_trapped_perCell # Remove number of adults trapped per cell from distribution
			self.adultReproAdult[traps_loc_index] = self.adultReproAdult[traps_loc_index] - N_adults_trapped_perCell # Remove number of adults trapped per cell from adult able to reproduce
			self.adults_trapped_loc[traps_loc_index] += N_adults_trapped_perCell # save spatial location of adults trapped
			self.nbAdults_trapped[simulationTime] = np.sum(N_adults_trapped_perCell) # count number of adults trapped
		# Trap for young adults (settled but not able to reproduce)
		if np.sum(self.youngAdultdistribution)>0  and len(traps_loc_index[0])>0:
			N_youngadults_trapped_perCell = [self.trapping_func(self.youngAdultdistribution[c,d], self.trap_density[c,d]) for c,d in zip(traps_loc_index[0],traps_loc_index[1])]  # Nb of subadults trapped per cell
			self.distribution[traps_loc_index] -= N_youngadults_trapped_perCell # Remove number of subadults trapped per cell from distribution
			self.youngAdultdistribution[traps_loc_index] -= N_youngadults_trapped_perCell # Remove number of subadults trapped per cell from adult able to reproduce
			self.adults_trapped_loc[traps_loc_index] += N_youngadults_trapped_perCell # save spatial location of subadults trapped
			self.nbAdults_trapped[simulationTime] += np.sum( N_youngadults_trapped_perCell) # count number of subadults trapped
	################# Main model
	def run(self):
		self.populate_predators() # initiate population inside and outside the treatment area
		index=0 
		for i in range(self.simulationTime):
			self.densityPerMonth[i] = np.sum(self.distribution[self.my_study_area>0]) # record number of individuals per month
			if i % self.timestep_perYear == 0:
				print('Time ',i,' - replication: ',self.nbrep)	
				### To save distribution in text file
				name_output_file = self.output_distribution + '/distribution_overTime_'+str(i)+'_rep_'+ str(self.nbrep) +'.gz'
				np.savetxt(name_output_file, self.distribution, fmt='%f', delimiter='\t')
				name_output_file = self.output_distribution + '/dispersion_pattern_'+str(i)+'_rep_'+ str(self.nbrep) +'.gz'
				np.savetxt(name_output_file, self.pattern_dispersion, fmt='%f', delimiter='\t')
				name_output_file = self.output_distribution + '/juveniles_tapped_'+str(i)+'_rep_'+ str(self.nbrep) +'.gz'
				np.savetxt(name_output_file, self.juveniles_trapped_loc, fmt='%f', delimiter='\t')
				name_output_file = self.output_distribution + '/adults_trapped_'+str(i)+'_rep_'+ str(self.nbrep) +'.gz'
				np.savetxt(name_output_file, self.adults_trapped_loc, fmt='%f', delimiter='\t')
			self.reproduce(i) # population dynamic (monhtly reproduction)
			if self.nbOffsprings>0: 
				self.move(i) # daily dispersal of offspring (if trapping occur, juveniles will be trapped here)
			self.mortality(i) # adult mortality
			if i % self.timestep_perYear in self.trapping_session: # Trapping for adults during active trapping session. 
				self.trap_adults(i)


#### Main function to run model and save outcomes
def runmodel(my_habitat_K, my_study_area, output_folder, rep, simulationTime, resolution, max_distance, predator, Fmin, Fmean, Fmax, life_span, gamma0, sigma, gamma1, habitat_dependence,density_dependence, initPopSize, my_trap_loc, nb_nights, trapping_frequency):
	seed = random.randint(0,1000)
	random.seed(seed)
	np.random.seed(seed)
	max_resolution = 2000 # Maximum resolution (velocity per day) 
	if isinstance(my_trap_loc, np.ndarray):
		my_trap_loc_str = 'proposedTrapline' 
	else:
		my_trap_loc_str = my_trap_loc
	# the variable parameter is used to save the outcomes of the simulations. One file per replication and parameter combinations.
	parameter = '_resolution_'+str(resolution)+'_maxDistance_'+str(max_distance)+'_Fmin_'+str(Fmin)+'_Fmean_'+str(Fmean)+'_Fmax_'+str(Fmax)+'_lifeSpan_'+str(life_span)+'_habitatD_'+str(habitat_dependence)+'_densityD_'+str(density_dependence)+'_g0_'+str(gamma0)+'_sigma_'+str(sigma)+'_g1_'+str(gamma1)+'_init_'+str(initPopSize)+'_TrapDensity_'+str(my_trap_loc_str)+'_trappingMonth_'+str(trapping_frequency)+'_nbNights_'+str(nb_nights)+'_rep_'+str(rep)
	# Create a folder to save distribution patterns
	output_distribution = output_folder + '/distribution' + parameter  # create a subfolder for distribution map
	if not os.path.exists(output_distribution):
		os.makedirs(output_distribution)
	# Initialise the model
	my_pop = envir(my_habitat_K, my_study_area, rep, resolution, max_resolution, predator, Fmin, Fmean, Fmax, life_span,gamma0, sigma, gamma1, max_distance, habitat_dependence, density_dependence, initPopSize, my_trap_loc,  nb_nights, trapping_frequency, simulationTime, output_distribution)
	name_output_habitat = output_folder  + '/0.carrying_capacity_map'+ parameter + '.gz' #save the carrying capacity map 
	np.savetxt(name_output_habitat, my_pop.resized_habitat, fmt='%f', delimiter='\t')
	# Run model
	my_pop.run()
	# Save model output
	name_output_file= output_folder + '/1.general_variables' + parameter + '.gz'
	# save initial parameters: pop size, carrying capacity, final population abundance, etc. for record
	Ktest = my_habitat_K[my_habitat_K>0]
	output_variable=np.column_stack((np.array(my_pop.initPopSize),np.array(np.sum(Ktest[Ktest>=0])),np.array(np.sum(my_pop.resized_habitat[my_pop.resized_habitat>=0])),np.array(np.sum(my_pop.resized_habitat[my_pop.my_study_area>0])),np.array(my_pop.resized_habitat.shape[0]),np.array(my_pop.resized_habitat.shape[1]),np.array(np.sum(my_pop.distribution)),np.array(simulationTime),np.array(resolution),np.array(max_distance),np.array(Fmin),np.array(Fmean),np.array(Fmax),np.array(life_span),np.array(habitat_dependence),np.array(density_dependence),np.array(initPopSize)))
	np.savetxt(name_output_file,output_variable, fmt='%f', delimiter='\t')
	# save monthly stat
	name_output_stat= output_folder + '/2.monthly_stat'  + parameter +'.gz'
	output_permonth=np.column_stack((np.array(my_pop.densityPerMonth),np.array(my_pop.nbAdults_reproducing),np.array(my_pop.nbBirthPerMonth),np.array(my_pop.dipsersalMortality),np.array(my_pop.settlingRate),np.array(my_pop.emigrationRate),np.array(my_pop.imigrationRate),np.array(my_pop.nbjuveniles_trapped),np.array(my_pop.nbAdults_death),np.array(my_pop.nbAdults_trapped)))
	np.savetxt(name_output_stat,output_permonth, fmt='%f', delimiter='\t')#, header=headermonthlystat)
	# save average distance (nb cell) and euclidean distance
	name_output_distance = output_folder + '/3.dispersal_distance'  + parameter + '.gz'
	np.savetxt(name_output_distance,my_pop.dispersal_distance, fmt='%f', delimiter='\t')


#### Main function to parallelize simulations
def main(parameter_combinations):
	# parallelism
	n_cpus = 2
	pool = multiprocessing.Pool(n_cpus)
	try:
		pool.starmap(runmodel, [params for params in parameter_combinations])
	finally:
		pool.close()



####################################### Read suitability map

# Read habitat suitability. Map should be a raster file saved in ASCII format. 
# Each raster cell countain a float (carrying capacity of the predator under study)
# The maps should be generated at a resolution 100m (or need to change the parameter of the function habitat_suitability.bin_ndarray)
filename_myhabitat =  '~/Data/Ascii_PossumGrid' # habitat suitability as provided by the user
my_init_landscape=habitat_suitability.readMap(filename_myhabitat) #  read map of shape (342, 281)
# Make sure dimension are odd number of 5 so that we can upscale it to a resolution of 500m
my_habitat_K=my_init_landscape[0:340,0:280]

####################################### Read study area boundaries

filename_studyArea =  '~/Data/Ascii_StudyArea' # boundraies of treatment area as provided by the user
my_study_area=habitat_suitability.readMap(filename_studyArea) # read map of shape (527, 407)
my_study_area_smallextent=my_study_area[0:340,0:280] # Update dimension

####################################### Read trap lines

filename_trap = '~/Data/raster_trapsNZ_location' # Network of traps - extracted from https://www.trap.nz/
my_trap_line = habitat_suitability.readMap(filename_trap) # read map of shape hape (527, 407)
my_trap_line_smallextent = my_trap_line[0:340,0:280]  # Update dimension

# Names of outpout folder where simulation outcomes will be saved
output_folder = '~/Results'# str(sys.argv[5])
if not os.path.exists(output_folder):
	os.makedirs(output_folder)



################### Define simulation parameter
nbrep = 2  # number of replication
simulationTime =  5 # simulation time in years
resolution = 500 # resolution in meters
species= 'possum'  # names of the species (can only be 'possum' at the moment)
# simulation scenario
max_dispersal= 12000 # maximum dispersal distance
min_birth = 0.5 # min birth
mean_birth= 0.7 # mean birth
max_birth = 1.02 # max birth
life_span = 12 # life sapn
init_pop = 500 # number of individuals in the landscape to initiate simulations
habitat_dependence = 1 # habitat dependence scaling factor
density_dependence = 1 # density dependence scaling factor
gamma0= 0.05 # Adult trapability index
sigma= 63 # half-normal trapability decay 
gamma1= 0.05 # trapability index
trap_density = [my_trap_line_smallextent] # define an homogenous density of trap over the landscape
my_trap_loc_str = 'proposedTrapline' 
number_of_night_trapping = [2] # number of nights trapping occurs during a trapping session
trapping_frequency= [7] # number of month trapping occur during the year


# Save the set of parameters
study_parameter= list(([output_folder],[nbrep],[simulationTime*12],resolution,max_dispersal,species, min_birth, mean_birth, max_birth,life_span,gamma0,sigma,gamma1,habitat_dependence, density_dependence,init_pop,my_trap_loc_str,number_of_night_trapping,trapping_frequency,[filename_myhabitat],[filename_studyArea], [filename_trap]))
filename_parameter= output_folder + '/0_Study_parameters.txt'
thefile_parameter = open(filename_parameter, 'w')
for item in study_parameter:
	a=thefile_parameter.write("%s\n" % item)

thefile_parameter.close()


# Parameter combinations
parameter_combinations = itertools.product([my_habitat_K],[my_study_area_smallextent], [output_folder],[k for k in range(nbrep)],[simulationTime], [resolution], [max_dispersal], [species], [min_birth],[mean_birth], [max_birth], [life_span], [gamma0], [sigma], [gamma1], [habitat_dependence], [density_dependence], [init_pop], trap_density, number_of_night_trapping, trapping_frequency)

# Each replication of the combination of parameter is independent of each other 
# Therefore each replication can be executed in a different CPU (# parallel execution of replications) 
main(parameter_combinations)

