# identification_modele_avion
identification_modele_avion collab A. LETALENET M. ALLICHE

Différents scirpts : 
Coef_aero_equation.ipynb = Décrit les différentes étapes pour générer les équations des 			   coefficients aérodynamiques Cd et Cl, ainsi que leur lambdification. 

Force_aero_equation.ipynb = Décrit les différentes étapes de génération des équations pour les 				    vecteurs des forces aérodynamiques dans le repère inertielle. Permet 			            aussi de lambdifier ces équations dans le cas de 5 surfaces portantes 			            et avec 4 moteurs tournant à la même vitesse. 

equation_generator.ipynb = Regroupe les différents scripts décrit ci-dessus (Attention, les 				   fichier ne sont pas liés directement!) et adapte les 			   équations pour une géométrie particulière. Génère ensuite les équations 				   lambdifier, et les exportent vers le fichier "function_moteur_physique"
		           tel que : 
    			# 0 : VelinLDPlane_function
   			# 1 : dragDirection_function
  			# 2 : liftDirection_function
  			# 3 : compute_alpha_sigma() = [alpha, sigma]
   			# 4 : Coeff_function = [Cl, Cd] :  renvoi les coefficients aéro 
    			# 5 : Effort_Aero = [Force, Couple] : renvoi un liste des efforts 
        

MoteurPhysique_class.py = Exploite les fonctions du fichier de fonctions produit par le script 				"equation_generator.ipynb", pour reconstruire la dynamique du drone. Il 			log aussi les données de vols dans un fichier "log.txt", et il sauvegarde 				aussi dans un autre fichier ("params.txt"), les grandeurs utilisées lors 				de la simulation. 

Gui_class.py = Cette classe génère les fenêtres graphiques qui vont permettre d'observer le drone 		       lors de la simulation en temps réel. De plus il gère aussi les entrées envoyées 		       depuis le joystick vers le moteurs physique. 

Simulator_Class.py = Cette classe fait le lien entre le moteur physique et la classe GUI qui gère 			     la partie graphique et les entrées. Dans un premier temps, on récupère les 		     entrées du joystick avec la classe GUI, qui sont ensuité injectées dans le 		     moteur physique qui va alors données les nouvelles accélération et 	 			     orientation qui servirons à alimenter l'interfaces graphiques. 



Pour lancer une simulation : 
 		- Etape 1 : lancer le scripts "equation_générator.ipynb" et vérifier que le fichier de fonctions a bien été écrit. 
	        - Etape 2 : brancher une manette et lancer le fichier "Simulator_Class.py", la simulation va se lancer et va se dérouler en 3 temps : 
	        		- Temps d'initialisation, les efforts sont nuls (par défault 5s) 
	        		- Le décollage, jusqu'à ce que le drone décolle les rotations sont suivant -x et -y sont bloquées. 
	        		- Au moment du décollage, la grille change de couleur. 
	        		- Vol en pilotage manuel
	        		
	Il est possible de changer les paramètres physiques du drones dans la classe du moteur physique (seules les valeurs des aires sont à modififier dans le script "equation_generator" et le relancer pour  générer les équations). Une fois fait, on peut relancer la simulation directement. 

Pour le pilotatge manuel : 
	- joystick gauche horizontale = roll
	- joystick gauche vertical = pitch
	- joystick droite horizontale = yaw
 	- joystick drotie vericale = vitesse des moteurs
Pour ajuster le comportement, on peut modifier les gains des ailerons de controle (k0, k1, k2) par exemple.



