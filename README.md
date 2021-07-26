# identification_modele_avion
identification_modele_avion collab A. LETALENET M. ALLICHE

Différents scirpts : 
equation_generator.ipynb = Regroupe les différents scripts décrit ci-dessus (Attention, les fichier ne sont pas liés directement!) et adapte les équations pour une géométrie particulière. Génère ensuite les équations lambdifier, et les exportent vers le fichier "function_moteur_physique" tel que : 
    		# 0 : VelinLDPlane_function
   			# 1 : dragDirection_function
  			# 2 : liftDirection_function
  			# 3 : compute_alpha en fonction des VelinLDPlane, dragDirection, et liftDirection
   			# 4 : Effort_Aero_complete_function = [Force, Couple] : renvoi un liste des efforts en fonction d'une liste de alpha ainsi que de la vitesse et orientation du drone, dans le repère Body
    		# 5 : Grad_Effort_Aero_complete_function = renvoi le gradient des forces, calculé à partir de [6], dans le repère body
    		# 6 : RMS_forces_function = Renvoi les erreurs au carrés des forces pour un jeu de données d'entrée, et des données de sorties normalisé
    		# 7 : RMS_torque_function = Renvoi les erreurs au carrés des couoles pour un jeu de données d'entrée, et des données de sorties normalisé 
    		# 8 : Cout_function = Calcul la fonction de cout, somme des RMS errors des couples et des forces normalisé. 
    		# 9 : Grad_Cout_function = Calcul le gradient de la fonction de cout.
    		# 10: theta = List de clefs renvoyant toutes les données utilisé pour l'identification. Tout les calculs sont fait à partir de cette liste, il est important que les noms utilisés soit les mêmes que ceux du Moteur Physique.


MoteurPhysique_class.py = Exploite les fonctions du fichier de fonctions produit par le script "equation_generator.ipynb", pour reconstruire la dynamique du drone. Il log aussi les données de vols dans un fichier "log.txt", et il sauvegarde aussi dans un autre fichier ("params.txt") les grandeurs utilisées lors de la simulation. De plus il est utilisé par l'optimizeur pour calculer la fonction de cout ainsi que son gradient. 

Gui_class.py = Cette classe génère les fenêtres graphiques qui vont permettre d'observer le drone lors de la simulation en temps réel. De plus il gère aussi les entrées envoyées depuis le joystick vers le moteurs physique. 

Simulator_Class.py = Cette classe fait le lien entre le moteur physique et la classe GUI qui gère la partie graphique et les entrées. Dans un premier temps, on récupère les entrées du joystick avec la classe GUI, qui sont ensuité injectées dans le moteur physique qui va alors données les nouvelles accélération et orientation qui servirons à alimenter l'interfaces graphiques. 

Optimizer_script.py : Ce script permet de lancer les différentes optimisation, une fonction permet de préparer les données, en découpant et en mélangeant les dataset. Une fonction permet de calculer le gradient suivant différentes méthodes (calcul symbolique(+rapide) se trouvant dans le moteur physique, calcul numérique(-rapide)). 


Pour lancer une simulation : 
 		- Etape 1 : lancer le scripts "equation_générator.ipynb" et vérifier que le fichier de fonction a bien été écrit. 
        - Etape 2 : brancher une manette et lancer le fichier "Simulator_Class.py", la simulation va se lancer et va se dérouler en 3 temps : 
        		- Temps d'initialisation, les efforts sont nuls (par défault 1s) 
        		- Le décollage, jusqu'à ce que le drone décolle les forces allant vers le sol sont bloqué, après les forces s'applique normalement quelque soit la position du drone. 
        		- Au moment du décollage, la grille change de couleur. 
        		- Vol en pilotage manuel avec la manette.


Il est possible de changer les paramètres physiques du drones dans la classe du moteur physique directement, les paramètres sont alors écrit à chaque simulation dans un fichier .json. 

Pour le pilotatge manuel : 
	- joystick gauche horizontale = roll
	- joystick gauche vertical = pitch
	- joystick droite horizontale = yaw
 	- joystick drotie vericale = vitesse des moteurs
Pour ajuster le comportement, on peut modifier les gains des ailerons de controle (k0, k1, k2) par exemple.


Pour lancer une optimisation : 
		- Etape 1 : Générer des données de logs, et choisir ces données dans l'optimizeur. 
		- Etape 2 : Réglé les valeurs initials des données à optimise, pour cela, avec des données simulées, on prend les vrais valeurs que l'on randomize, il est possible de les choisr manuellement si souhaiter.
		- Etape 3 : Réglé les différents gain du PID, ainsi que le batch size, et le nombre d'epoch, et lancer l'optimisation. 2 fenêtres vont s'ouvrir, la premieres permet de voir les erreurs en % en fonction des epochs, et la secondes permet de voir l'évolution de la qualité de la simulation au fur et mesure que l'otpimisation avance. Pour cela elle relance une simulation avec les mêmes entrées, mais à chaque fois avec les paramètres courant, et cela permet de retracer la qualité de la simu. 

