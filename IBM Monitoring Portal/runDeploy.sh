#!/bin/bash
#!/usr/bin/expect
# A script for deploying an RMS (Regional Monitoring Server) according to one of the schemes - ITM or ITNM.
# If the script is run without parameters, the schema is taken from the server name.
# If the script is run with the "SELECTED" parameter, selective installation of components is performed.
#
# The script should run with the following command:
# 	./runDeploy.sh | tee -a deploy.log
# (writing the operation log to the deploy.log file and displaying it at the same time)
#
echo
echo "========================================================"
echo " PFR MONITORING DEPLOY SCRIPT for Linux 6.5 (Version 5) "
echo "========================================================"
echo
#
#############################################################################################################################################################################
# ITM components:																																							#
# ==============    		     		                                                                  		                                                            #
# NAME				VERSION			COMPONENTS					DISTRIBUTION     								AGENT CODE                                     				#
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# DB2				10.01.00.00		Database                    DB2_ESE_10_Linux_x86-64.tar.gz   			                                                                #
# DB2_FP4			10.01.00.04		                            v10.1fp4_linuxx64_universal_fixpack.tar.gz                                                                  #
# ITM_FP2  			06.30.02.00		Server components 			ITM_V6.3.0.2_BASE_LNX64_EN.tar.gz				cq - Tivoli Enterprise Portal Server  						#
#					06.30.02.00		and base agents																hd - Warehouse Proxy										#
#					06.30.02.00																					kf - IBM Eclipse Help Server								#
#					06.30.02.00																					pa - Performance Analyzer                                   #
#					06.30.02.00																					sy - Summarization and Pruning Agent						#
# ITM_FP5			06.30.05.00		Server components 			6.3.0-TIV-ITM_TMV-Linuxx64-FP0005.tar.gz		cq - Tivoli Enterprise Portal Server  						#
#					06.30.05.00		and base agents																hd - Warehouse Proxy										#
#					06.30.05.00																					kf - IBM Eclipse Help Server								#
#					06.30.05.00																					pa - Performance Analyzer                                   #
#					06.30.05.00																					sy - Summarization and Pruning Agent						#
# OS_FP2 			06.30.02.00		Agent and Support			ITM_V6.3.0.2_AGT_MP_ML.tar.gz					lz - Monitoring Agent for Linux OS                          #
#       			06.30.02.00		Agent and Support															a4 - Monitoring Agent for i5/OS                             #
#       			06.30.02.00		Agent and Support															nt - Monitoring Agent for Windows OS                        #
#       			06.30.02.00		Support																		ux - Monitoring Agent for UNIX OS                           #
# OS_FP5			06.30.05.00		Agent and Support			6.3.0-TIV-ITM_TMV-Agents-FP0005.tar.gz			lz - Monitoring Agent for Linux OS                          #
#       			06.30.05.00		Agent and Support															a4 - Monitoring Agent for i5/OS                             #
#       			06.30.05.00		Agent and Support															nt - Monitoring Agent for Windows OS                        #
#       			06.30.05.00		Support																		ux - Monitoring Agent for UNIX OS                           #
# OS_FP5_IF2		06.30.05.02		Support						6.3.0.5-TIV-ITM_LINUX-IF0002.tar				lz - Monitoring Agent for Linux OS                          #
# DB2   			07.10.00.00		Agent and Support			ITCAM_APPS_7.1_AG_FOR_DB2.tar.gz				ud - IBM Tivoli Composite Application Manager Agent for DB2	#
# WAS72     		07.20.00.00		Agent and Support			agent_was.zip   								yn - ITCAM Agent for WebSphere Applications                 #
# WAS72_FP1     	07.20.00.01		Agent and Support			7.2.0.0-TIV-ITCAMAD_WS_Linuxx-IF0001.tar   		yn - ITCAM Agent for WebSphere Applications                 #
# VMWareVI		 	07.20.01.01		Agent and Support			ITMVE7.2.0.2VMVIKVMNAS_AGTSUP.tar.gz			vm - Monitoring Agent for VMware VI      		            #
### VMWareVI_FP2	07.20.02.01	    Agent and Support			7.2.0.2-TIV-ITM_VMWVI-IF0001.tar				vm - Monitoring Agent for VMware VI			        	    #
# VMWareVI_FP4		07.20.04.00	    Agent and Support			ITMVE7.2.0.3_VMVIKVMNAS_AGTMP_EN.tar.gz			vm - Monitoring Agent for VMware VI			                #
# Transactions		07.40.00.00	    Agent and Support			ITCAMT_V7.4.0.0.1_RT_-_LIN_MP,_EN.tar			t3 - ITCAM Console								            #
#    				07.40.00.00	    Agent and Support															t6 - ITCAM for Robotic Response Time    		            #
# Transactions		07.40.01.00	    Agent and Support			ITCAM_FOR_TRAN_7.4.0.1_RT_LINX_EN.tar			t3 - ITCAM Console								            #
#    				07.40.01.00	    Support																		t6 - ITCAM for Robotic Response Time    		            #
# ISM 				07.40.00.13	    Agent and Support			ITCAMforTransactions.tar						is - ITCAM for Transactions: Internet Service Monitoring    #
# Storage Manager   06.34.00.00		Support						6.3.4.000-TIV-TSMRPT-AGENT-Linux.bin			sk - Monitoring Agent for Tivoli Storage Manager            #
# HTTP 		    	07.10.03.00		Support						agent_http.zip							   		ht - IBM Tiv. Composite App. Manager Agent for HTTP Servers #
# HTTP_FP3 		    07.10.03.00		Support					    7.1.0-TIV-ITCAMAD_HTTP_Linuxx-FP0003.tar	   	ht - IBM Tiv. Composite App. Manager Agent for HTTP Servers #
# IIS 		        06.30.00.00		Support					    ITCAM_FOR_MSAPPS_V6.3_ADV_CS.tar.gz			   	q7 - Microsoft Internet Information Services (IIS) Agent    #
# MQ	     		07.30.01.00		Support						ITCAM_AGENTS_WS_MSG_7.3_MF.tar.gz	    		mq - WebSphere MQ Monitoring Agent                          #
# MQ FTE    		07.01.00.00		Support						CZJQ7EN.tar							    		m6 - ITCAM Agent for WebSphere MQ File Transfer Edition     #
# Network Devices   06.22.00.00		Support						agent_net.zip   								n4 - Monitoring Agent for Network Devices					#
# Network Manager	03.90.01.00		Support 					ITMagent.tar.gz		  						    np - IBM Tivoli Network Manager					            #
# Systems Director	06.22.00.00		Support 					CZF82EN.tar.gz   	  						    d9 - Monitoring Agent for IBM Systems Director base         #
# TSPC				06.30.05.00		Support 					agent_TSPC.zip   	  						    p1 - Monitoring Agent for TPC                               #
# CACHE     		06.21.00.00		Support 					K34.tgz								     		34 - Monitoring Agent for Cache PFR                         #
# Log File Agent	06.30.01.00		Support 					6.3.0-TIV-ITM_LFA-FP0001.tar.gz					lo - Tivoli Log File Agent		                            #
#                   				                                                                       		                                                            #
#############################################################################################################################################################################
# ITNM components:																																							#
# ===============	   		     		                                                                  		                                                            #
# NAME				VERSION			COMPONENTS					DISTRIBUTION     								AGENT CODE                                     				#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# DB2				10.01.00.00		Database                    DB2_ESE_10_Linux_x86-64.tar.gz   			                                                                #
# DB2_FP4			10.01.00.04		                            v10.1fp4_linuxx64_universal_fixpack.tar.gz                                                                  #
# ITNM				4.1.1.11	    Server components			IBM_TIVOLI_NETWORK_MGR_IP_EDI_LNX.tar																		#
# Netcool/OMNIbus 	7.4.0.0			Server components			NC_OMNI_CORE_74_LIN.tar			    	        															#
# EIF-probe							Server components			NC-OMNI_PRB_F_TIV_EIF_FOR_LINX_EN.tar.gz        															#
# Event Synchronization				Server components			tools.zip                       																			#
# OS_FP2 			06.30.02.00		Agent and Support			ITM_V6.3.0.2_AGT_MP_ML.tar.gz					lz - Monitoring Agent for Linux OS                          #
#       			06.30.02.00		Agent and Support															a4 - Monitoring Agent for i5/OS                             #
#       			06.30.02.00		Agent and Support															nt - Monitoring Agent for Windows OS                        #
#       			06.30.02.00		Support																		ux - Monitoring Agent for UNIX OS                           #
# OS_FP5			06.30.05.00		Agent and Support			6.3.0-TIV-ITM_TMV-Agents-FP0005.tar.gz			lz - Monitoring Agent for Linux OS                          #
#       			06.30.05.00		Agent and Support															a4 - Monitoring Agent for i5/OS                             #
#       			06.30.05.00		Agent and Support															nt - Monitoring Agent for Windows OS                        #
#       			06.30.05.00		Support																		ux - Monitoring Agent for UNIX OS                           #
# Network Devices   06.22.00.00		Agent						agent_net.zip   								n4 - Monitoring Agent for Network Devices					#
# Network Manager	03.90.01.00		Agent	 					ITMagent.tar.gz		  						    np - IBM Tivoli Network Manager					            #
# Universal			06.23.05.00		Agent						um_062305000_LINUX.tar.gz						um - Universal Agent										#
#	MySQL 							Config 						mysql.zip																									#
# 	PTK 							Config  																																#
# 	Storwize						Config  																																#
# 	UPS								Config  																																#
#                   				                                                                       		                                                            #
#############################################################################################################################################################################
#
BLOCK_FINISH_PAUSE=5
DISTRIB_HOME=''
SCRIPT_HOME=''
#
a=$(hostname)
HOST=${a::${#a}-3}
REGION_CODE=${a#*M}
#
case "$HOST" in
	"ITM")
		if [ "$1" == "SELECTED" ]
			then
				DEPLOY_SCHEME='SELECTED'
			else
				DEPLOY_SCHEME='ITM'
		fi
		;;
	"ITNM")
		if [ "$1" == "SELECTED" ]
			then
				DEPLOY_SCHEME='SELECTED'
			else
				DEPLOY_SCHEME='ITNM'
		fi
		;;
	*)	
		echo -e "\a" 
		echo "ERROR"
		echo
		echo "========================================================================"
		echo " PFR MONITORING DEPLOY SCRIPT SHOULD BE RUN on ITMxxx or ITNMxxx SERVER "
		echo "========================================================================"
		echo
		exit 2
esac
#
cd $SCRIPT_HOME
echo "-----------------------------------------------------------------------"
echo -n "Current date is:               "; date
echo "-----------------------------------------------------------------------"
echo "Disk space:"
df / -h
FREE_SPACE=$(df / -m | awk '{ if (NR > 1)  print $4}')
echo "-----------------------------------------------------------------------"
echo "The \$REGION_CODE is:          "    $REGION_CODE
echo "-----------------------------------------------------------------------"
echo "The \$DEPLOY_SCHEME is:        "    $DEPLOY_SCHEME
echo "-----------------------------------------------------------------------"
echo
echo "Selected deploy parameters:    "
case "$DEPLOY_SCHEME" in
	"ITM")
	cat $SCRIPT_HOME/runDeploy.sh | grep -A45 "# ITM scheme parameters" | grep -v "cat" | grep "'y'" | awk '{ print "\t\t\t       "$1 }'
	;;
	"ITNM")
	cat $SCRIPT_HOME/runDeploy.sh | grep -A22 "# ITNM scheme parameters" | grep -v "cat" | grep "'y'" | awk '{ print "\t\t\t       "$1 }'
	;;
	"SELECTED")
	cat $SCRIPT_HOME/runDeploy.sh | grep -A53 "# All installation parameters" | grep -v "cat" | grep "'y'" | awk '{ print "\t\t\t       "$1 }'
	;;
esac
echo "-----------------------------------------------------------------------"
sleep 5
#
# All installation parameters
#
CHECKMD5SUM='n' 						#check distributives integrity
EXTRACT_DISTRIBS='n' 					#extract distributives
INSTALL_DB2='n' 						#install DB2
CONFIG_DB2='n' 							#configure DB2
INSTALL_DB2FP='n' 						#install DB2 Fix Pack
INSTALL_ITNM='n' 						#install ITNM
CONFIG_ITNM='n' 						#configure ITNM for autostart and omni.dat
INSTALL_EIF='n' 						#install EIF
CONFIG_EIF='n' 							#configure EIF
INSTALL_ESYNC='n' 						#install ESYNC
CONFIG_ESYNC_AND_COLLECTION_LAYER='n'	#configure ESYNC and COLLECTION LAYER
CONFIG_GATE_UNI='n' 					#configure GATEWAY
CONFIG_NCO_PA='n' 						#configure Process Agent
DEPLOY_ITMBASE='n' 						#deploy TEMS, TEPS & other base components
CONFIG_HDAGENT='n' 						#configure Warehouse Proxy agent
CONFIG_SYAGENT='n' 						#configure Summarization and Pruning agent
DEPLOY_LZAGENT='n' 						#install and configure agent for operating system
DEPLOY_UDAGENT='n' 						#install and configure agent for DB2
DEPLOY_YNAGENT='n' 						#install and configure agent for WAS
INSTALL_VMAGENT='n' 					#install agent for VmWare VI 
#INSTALL_VMFPAGENT='n' 					#install agent Fix Pack for VmWare VI 
CONFIG_VMAGENT='n' 						#configure agent for VmWare VI 
INSTALL_T3AGENT='n' 					#install agent for ITCAM Console
INSTALL_ISMAGENT='n' 					#install agent for Internet Service Monitoring
DEPLOY_N4AGENT='n' 						#install and configure agent for Network Manager 
DEPLOY_NPAGENT='n' 						#install and configure agent for Netcool Precision
DEPLOY_UMAGENT='n' 						#install and configure Universal agent
DEPLOY_T6SUPPORTS='n' 					#install supports for ITCAM for Robotic Response Time agent
DEPLOY_SKSUPPORTS='n' 					#install supports for Storage Manager agent
DEPLOY_HTSUPPORTS='n' 					#install supports for HTTP agent
DEPLOY_HTFPSUPPORTS='n' 				#install supports for HTTP agent Fix Pack
DEPLOY_Q7SUPPORTS='n' 					#install supports for IIS agent
DEPLOY_MQSUPPORTS='n' 					#install supports for MQ agent
DEPLOY_MQFTESUPPORTS='n' 				#install supports for MQ FTE agent
DEPLOY_N4SUPPORTS='n' 					#install supports for Network Manager agent
DEPLOY_NPSUPPORTS='n' 					#install supports for NetCool Precision agent
DEPLOY_D9SUPPORTS='n' 					#install supports for Systems Director agent
DEPLOY_P1SUPPORTS='n' 					#install supports for Storage productivity center agent
DEPLOY_LOSUPPORTS='n' 					#install supports for Log File agent
REBUILD_TEMS='n' 						#rebuild TEMS
REBUILD_TEPS='n' 						#rebuild TEPS
RESTART_ITM='n' 						#restart all components and turn on SDA
DEPLOY_K34SUPPORTS='n' 					#install supports for CACHE agent
CONFIG_N4DEVICES='n' 					#configure Network devices for Network Manager agent
DISABLE_STANDARD_SITUATIONS='n' 		#disable standard situations
# IMPORT_SITUATIONS='n' 					#import custom situations
# DISTRIBUTION_SITUATIONS='n' 				#distribution situations 
# ASSOCIATION_SITUATION='n' 				#association situations 
CONFIGURE_HISTORY='n' 					#configure history collections
CREATE_OPERATOR_USER='n' 				#create new user for ITM (operator)
CUSTOMIZE_WORKSPACES='n' 				#customize TEPS workspaces
NCO_CHECK='n' 							#check NCO operability
FINAL_REBOOT='n' 						#reboot after installation
#
# ITM scheme parameters
#
if [ $DEPLOY_SCHEME == "ITM" ] 
	then
		CHECKMD5SUM='y' 
		EXTRACT_DISTRIBS='y' 
		INSTALL_DB2='y' 
		CONFIG_DB2='y' 
		INSTALL_DB2FP='y' 
		DEPLOY_ITMBASE='y' 
		CONFIG_HDAGENT='y' 
		CONFIG_SYAGENT='y' 
		DEPLOY_LZAGENT='y' 
		DEPLOY_UDAGENT='y' 
		DEPLOY_YNAGENT='y'
		INSTALL_VMAGENT='y' 	
		#INSTALL_VMFPAGENT='y' 	
		CONFIG_VMAGENT='y'
		INSTALL_T3AGENT='y' 	
		INSTALL_ISMAGENT='y' 
		DEPLOY_T6SUPPORTS='y' 
		DEPLOY_SKSUPPORTS='y' 
		DEPLOY_HTSUPPORTS='y' 
		DEPLOY_HTFPSUPPORTS='y' 
		DEPLOY_Q7SUPPORTS='y'
		DEPLOY_MQSUPPORTS='y' 
		DEPLOY_MQFTESUPPORTS='y' 
		DEPLOY_N4SUPPORTS='y' 
		DEPLOY_NPSUPPORTS='y' 
		DEPLOY_D9SUPPORTS='y'
		DEPLOY_P1SUPPORTS='y'
		DEPLOY_LOSUPPORTS='y' 
		REBUILD_TEMS='y' 
		REBUILD_TEPS='y' 
		RESTART_ITM='y' 
		DEPLOY_K34SUPPORTS='y' 
		CONFIG_N4DEVICES='y'
		DISABLE_STANDARD_SITUATIONS='y' 
		# IMPORT_SITUATIONS='y' 
		# DISTRIBUTION_SITUATIONS='y'
		# ASSOCIATION_SITUATION='y' 		
		CONFIGURE_HISTORY='y' 		 
		CREATE_OPERATOR_USER='y' 
		CUSTOMIZE_WORKSPACES='y' 
		FINAL_REBOOT='y'
fi
#
# ITNM scheme parameters
#
if [ $DEPLOY_SCHEME == "ITNM" ]
	then
		CHECKMD5SUM='y' 
		EXTRACT_DISTRIBS='y' 
		INSTALL_DB2='y' 
		CONFIG_DB2='y' 
		INSTALL_DB2FP='y' 
		INSTALL_ITNM='y' 
		CONFIG_ITNM='y' 
		INSTALL_EIF='y' 
		CONFIG_EIF='y' 
		INSTALL_ESYNC='y' 
		CONFIG_ESYNC_AND_COLLECTION_LAYER='y' 
		CONFIG_GATE_UNI='y' 
		CONFIG_NCO_PA='y' 
		DEPLOY_LZAGENT='y' 
		DEPLOY_N4AGENT='y' 
		DEPLOY_NPAGENT='y' 
		DEPLOY_UMAGENT='y'		
		NCO_CHECK='y'
		FINAL_REBOOT='y'
fi
#
#################################################################################################################################################################################
#
if [ $CHECKMD5SUM == "y" ]
	then
		$SCRIPT_HOME/checkmd5.sh $DEPLOY_SCHEME
		if (( $? > 0 ))
			then
				echo -e "\nCopy valid distrib(s) and try again!\n"
				exit 1
			else
				sleep $BLOCK_FINISH_PAUSE
		fi
fi
#
if [ $EXTRACT_DISTRIBS == "y" ]
	then
		$SCRIPT_HOME/extract.sh $DEPLOY_SCHEME
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $INSTALL_DB2 == "y" ]
	then
		echo
		echo "======================================================================="
		echo "INSTALLING DB2"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.rsp/db2ese_101.rsp .
		$DISTRIB_HOME/db2/ese/db2setup -r $SCRIPT_HOME/db2ese_101.rsp
		rm -f db2ese_101.rsp
		if tac $SCRIPT_HOME/deploy.log | grep -B1000 "INSTALLING DB2" | grep -q "The execution completed successfully."
			then
				echo -e "\nEnd of installing DB2\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nDB2 installation failed!\n"
				exit 1
		fi
fi
#
if [ $CONFIG_DB2 == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CONFIGURING DB2"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.configs/db2/db2server /etc/rc.d/init.d
		chkconfig --add db2server		
		echo -e "\nDone.\n" 
		echo -e "\nEnd of configuring DB2\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $INSTALL_DB2FP == "y" ]
	then
		echo
		echo "======================================================================="
		echo "INSTALLING DB2 FIX PACK"
		echo "======================================================================="
		echo
		/home/db2inst1/sqllib/db2profile
		su - db2inst1 -c "db2stop force"
		su - dasusr1 -c "db2admin stop"
		su - db2inst1 -c "db2 terminate"
		#
		cd $DISTRIB_HOME/db2fp/universal/
		echo "no" | ./installFixPack -b /opt/ibm/db2/V10.1/
		if (( $? == 0 ))
			then 
				cd $SCRIPT_HOME
				su - dasusr1 -c "db2admin start";
				su - db2inst1 -c "db2start";
				su - db2inst1 -c "db2level"; 
				echo -e "\nEnd of installing DB2 Fix Pack\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nDB2 Fix Pack installation failed!\n"
				exit 1
		fi
fi
#
if [ $INSTALL_ITNM == "y" ]
	then
		echo
		echo "======================================================================="
		echo "INSTALLING ITNM"
		echo "======================================================================="
		echo
		# fix multicast
		/sbin/route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0
		#
		groupadd ncoadmin
		usermod -a -G ncoadmin root
		#
		useradd -G db2iadm1,dasadm1 -m ncim
		echo -e "password\npassword" | (passwd ncim)
		su - db2inst1 -c "db2start"
		#
		chmod o+x $DISTRIB_HOME/itnm/PrecisionIP/scripts/create_db2_database.sh
		su - db2inst1 -c "$DISTRIB_HOME/itnm/PrecisionIP/scripts/create_db2_database.sh NCIM ncim"
		chmod o+x $DISTRIB_HOME/itnm/PrecisionIP/scripts/catalog_db2_database.sh
		su - db2inst1 -c "$DISTRIB_HOME/itnm/PrecisionIP/scripts/catalog_db2_database.sh ITNM $DEPLOY_SCHEME$REGION_CODE 50000"
		#
		cp -f $SCRIPT_HOME/default.rsp/ITNM-silent-install.txt .
		#
		sed "125s@PACKAGE.DIR.NCO=--UserInput--@PACKAGE.DIR.NCO=$DISTRIB_HOME/OMNI74@" -i ITNM-silent-install.txt
		sed "153s@IAGLOBAL_OBJECTSERVER_PRIMARY_NAME=--UserInput--@IAGLOBAL_OBJECTSERVER_PRIMARY_NAME=NCO$REGION_CODE@" -i ITNM-silent-install.txt
		sed "239s@IAGLOBAL_PRECISION_DOMAIN0=--UserInput--@IAGLOBAL_PRECISION_DOMAIN0=ITNMD$REGION_CODE@" -i ITNM-silent-install.txt
		#
		$DISTRIB_HOME/itnm/install.sh -i silent -f $SCRIPT_HOME/ITNM-silent-install.txt
		#
		rm -f ITNM-silent-install.txt
		if tac $SCRIPT_HOME/deploy.log | grep -B1000 "INSTALLING ITNM" | grep -q "The TERMINATE command completed successfully."
			then
				echo -e "\nEnd of installing ITNM\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nITNM installation failed!\n"
				exit 1
		fi
fi
#
if [ $CONFIG_ITNM == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CONFIGURING ITNM"
		echo "======================================================================="
		echo
		sed "131s@LOCATION_FILE=\${NCHOME}\/etc\/itnm.cfg@LOCATION_FILE=\/opt\/IBM\/tivoli\/netcool\/etc/itnm.cfg@" -i /opt/IBM/tivoli/netcool/precision/bin/itnm_control_functions.sh
		#
		sed "69s@100000@600000@" -i /opt/IBM/tivoli/netcool/etc/precision/CtrlServices.cfg
		sed "71s@ 5@ 30@" -i /opt/IBM/tivoli/netcool/etc/precision/CtrlServices.cfg
		#
		echo "#
# omni.dat file as prototype for interfaces file
#
# Ident: $Id: omni.dat 1.5 1999/07/13 09:34:20 chris Development $
#
[NCO$REGION_CODE]
{
	Primary:	$DEPLOY_SCHEME$REGION_CODE 4100
}
[MSK_NCO_OS]
{
	Primary: nco-main 4100
}
[MSK_TBSM_OS]
{
	Primary: tbsm-main 4100
}
[NCO_GATE]
{
	Primary:	$DEPLOY_SCHEME$REGION_CODE 4300
}
[MSK_NCO_PA]
{
	Primary: nco-main 4200
}
[MSK_TBSM_PA]
{
	Primary: tbsm-main 4200
}
[NCO_PA]
{
	Primary:	$DEPLOY_SCHEME$REGION_CODE 4200
}

[NCO_PROXY]
{
	Primary:	$DEPLOY_SCHEME$REGION_CODE 4400
}" > /opt/IBM/tivoli/netcool/etc/omni.dat
		#
		/opt/IBM/tivoli/netcool/bin/nco_igen
		#
		cp -f $SCRIPT_HOME/default.configs/host/Netcool.sh /etc/profile.d
		#
		/opt/IBM/tivoli/netcool/precision/bin/itnm_start nco
		sleep 30
		/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password 'password' -server NCO$REGION_CODE < $SCRIPT_HOME/default.scripts/pfrCustFields4OS.sql
		/opt/IBM/tivoli/netcool/precision/bin/itnm_stop nco
		#
		echo -e "\nDone.\n"
		echo -e "\nEnd of configuring ITNM\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $INSTALL_EIF == "y" ]
	then
		echo
		echo "======================================================================="
		echo "INSTALLING EIF PROBE"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.rsp/responseEIF.txt .
		sed "30s@PROBE_OR_GATE_LOCATION=--UserInput--@PROBE_OR_GATE_LOCATION=$DISTRIB_HOME/EIF@" -i responseEIF.txt
		/opt/IBM/tivoli/netcool/omnibus/install/nco_install_integration -i SILENT -f $SCRIPT_HOME/responseEIF.txt
		rm -f responseEIF.txt
		#
		echo -e "\nDone.\n"
		echo -e "\nEnd of installing EIF Probe\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $CONFIG_EIF == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CONFIGURING EIF PROBE"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.configs/eif/tivoli_eif.props .
		sed "88s@--UserInput--@NCO$REGION_CODE@" -i tivoli_eif.props
		cp -f tivoli_eif.props /opt/IBM/tivoli/netcool/omnibus/probes/linux2x86
		rm -f tivoli_eif.props
		#
		cp -f $SCRIPT_HOME/default.configs/eif/itm_event.rules /opt/IBM/tivoli/netcool/omnibus/probes/linux2x86
		cp -f $SCRIPT_HOME/default.configs/eif/pfr_service_itm.rules /opt/IBM/tivoli/netcool/omnibus/probes/linux2x86
		cp -f $SCRIPT_HOME/default.configs/eif/tivoli_eif.rules /opt/IBM/tivoli/netcool/omnibus/probes/linux2x86
		cp -f $SCRIPT_HOME/default.configs/eif/itm_custom_override.rules /opt/IBM/tivoli/netcool/omnibus/probes/linux2x86
		#
		echo -e "\nDone.\n"
		echo -e "\nEnd of configuring EIF Probe\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $INSTALL_ESYNC == "y" ]
	then
		echo
		echo "======================================================================="
		echo "INSTALLING ESYNC"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.rsp/responseESYNC.txt .
		sed "234s@<value>@ITM$REGION_CODE@" -i responseESYNC.txt
		$DISTRIB_HOME/tools/tec/ESync2300Linux.bin -options $SCRIPT_HOME/responseESYNC.txt -silent
		rm -f responseESYNC.txt
		#
		echo -e "\nDone.\n"
		echo -e "\nEnd of installing ESYNC\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $CONFIG_ESYNC_AND_COLLECTION_LAYER == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CONFIGURING ESYNC AND COLLECTION LAYER"
		echo "======================================================================="
		echo
		/opt/IBM/tivoli/netcool/precision/bin/itnm_start nco
		sleep 30
		#
		/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password 'password' -server NCO$REGION_CODE < $SCRIPT_HOME/default.scripts/itm_proc.sql
		/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password 'password' -server NCO$REGION_CODE < $SCRIPT_HOME/default.scripts/itm_db_update.sql
		/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password 'password' -server NCO$REGION_CODE < $SCRIPT_HOME/default.scripts/deduplication.sql
		/opt/IBM/tivoli/netcool/omnibus/bin/nco_sql -user root -password 'password' -server NCO$REGION_CODE < $SCRIPT_HOME/default.scripts/pfr_collection.sql
		#
		echo -e "\nEnd of configuring ESYNC and collection layer\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $CONFIG_GATE_UNI == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CONFIGURING GATEWAY UNIDIRECTIONAL"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.configs/gate/objserv_uni.props .
		sed "98s@--UserInput--@NCO$REGION_CODE@" -i objserv_uni.props
		sed "107s@--UserInput--@u$REGION_CODE@" -i objserv_uni.props
		cp -f objserv_uni.props /opt/IBM/tivoli/netcool/omnibus/gates/objserv_uni
		rm -f objserv_uni.props
		#
		cp -f $SCRIPT_HOME/default.configs/gate/objserv_uni.map /opt/IBM/tivoli/netcool/omnibus/gates/objserv_uni
		cp -f $SCRIPT_HOME/default.configs/gate/objserv_uni.reader.tblrep.def /opt/IBM/tivoli/netcool/omnibus/gates/objserv_uni
		#
		echo -e "\nDone.\n"
		echo -e "\nEnd of configuring gateway unidirectional\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $CONFIG_NCO_PA == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CONFIGURING PROCESS AGENT"
		echo "======================================================================="
		echo
		cp -f $SCRIPT_HOME/default.configs/pa/nco_pa.conf .
		sed "12s@--UserInput--@NCO$REGION_CODE@" -i nco_pa.conf
		sed "43s@--UserInput--@NCO$REGION_CODE@" -i nco_pa.conf
		sed "13s@--UserInput--@ITNM$REGION_CODE@" -i nco_pa.conf
		sed "23s@--UserInput--@ITNM$REGION_CODE@" -i nco_pa.conf
		sed "34s@--UserInput--@ITNM$REGION_CODE@" -i nco_pa.conf
		sed "44s@--UserInput--@ITNM$REGION_CODE@" -i nco_pa.conf
		sed "89s@--UserInput--@ITNM$REGION_CODE@" -i nco_pa.conf
		cp -f nco_pa.conf /opt/IBM/tivoli/netcool/omnibus/etc
		rm -f nco_pa.conf
		#
		echo -e "\nDone.\n"
		echo -e "\nEnd of configuring Process Agent\n"
		echo -e "-----------------------------------------------------------------------\n"
		sleep $BLOCK_FINISH_PAUSE
fi
#
if [ $DEPLOY_ITMBASE == "y" ]
	then
		echo
		echo "======================================================================="
		echo "DEPLOYING ITM BASE COMPONENTS"
		echo "======================================================================="
		echo
		su - db2inst1 -c "db2start"
		su - db2inst1 -c "db2 catalog tcpip node MSKWH remote 10.103.0.58 server 50000"
		su - db2inst1 -c "db2 catalog database WH$REGION_CODE at node MSKWH"
		#
		useradd -m sysadmin
		echo -e "password\npassword" | (passwd sysadmin)
		#
		cp -f $SCRIPT_HOME/default.rsp/ms_silent_install_6302.txt .
		sed "113s@--UserInput--@HUB$REGION_CODE@" -i ms_silent_install_6302.txt
		$DISTRIB_HOME/base/install.sh -q -h /opt/IBM/ITM -p $SCRIPT_HOME/ms_silent_install_6302.txt
		rm -f ms_silent_install_6302.txt
		#
		cp -f $SCRIPT_HOME/default.rsp/ms_silent_install_6305.txt .
		sed "113s@--UserInput--@HUB$REGION_CODE@" -i ms_silent_install_6305.txt
		$DISTRIB_HOME/base_FP5/6.3.0-TIV-ITM_TMV-Linuxx64-FP0005/install.sh -q -h /opt/IBM/ITM -p $SCRIPT_HOME/ms_silent_install_6305.txt
		rm -f ms_silent_install_6305.txt
		#
		cp -f $SCRIPT_HOME/default.rsp/ms_silent_config_6302.txt .
		sed "223s@--UserInput--@ITNM$REGION_CODE@" -i ms_silent_config_6302.txt
		/opt/IBM/ITM/bin/itmcmd config -S -t HUB$REGION_CODE -p $SCRIPT_HOME/ms_silent_config_6302.txt
		rm -f ms_silent_config_6302.txt
		#
		cp -f $SCRIPT_HOME/default.rsp/cq_silent_config_6305.txt .
		sed "38s@--UserInput--@ITM$REGION_CODE@" -i cq_silent_config_6305.txt
		sed "144s@--UserInput--@u$REGION_CODE@" -i cq_silent_config_6305.txt
		sed "148s@--UserInput--@WH$REGION_CODE@" -i cq_silent_config_6305.txt
		sed "178s@--UserInput--@password@" -i cq_silent_config_6305.txt
		/opt/IBM/ITM/bin/itmcmd config -A -p $SCRIPT_HOME/cq_silent_config_6305.txt cq
		rm -f cq_silent_config_6305.txt
		#
		cp -f $SCRIPT_HOME/default.configs/host/ITM.sh /etc/profile.d
		sed -e "s/KT1_TEMS_SECURE=NO/KT1_TEMS_SECURE=YES/g" -i /opt/IBM/ITM/config/ms.ini
		echo "MAX_CTIRA_RECURSIVE_LOCKS=150" >> /opt/IBM/ITM/config/ms.ini
		echo "KDEP_SERVICETHREADS=63" >> /opt/IBM/ITM/config/ms.ini
		mkdir /opt/pfr_depot
		echo "DEPOTHOME=/opt/pfr_depot" >> /opt/IBM/ITM/config/kbbenv.ini
		#
		if tac $SCRIPT_HOME/deploy.log | grep -B2000 "DEPLOYING ITM BASE COMPONENTS" | grep -q "... InstallPresentation.sh completed"
			then
				echo -e "\nEnd of deploying ITM base components\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nITM base components installation failed!\n"
				exit 1
		fi
fi
#
#################################################################################################################################################################################
#
if [ $CONFIG_HDAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/HD/hd_configure.sh $REGION_CODE || exit 1
fi
#
if [ $CONFIG_SYAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/SY/sy_configure.sh $REGION_CODE || exit 1
fi
#
#################################################################################################################################################################################
#
if [ $DEPLOY_LZAGENT == "y" ]
	then 
		$SCRIPT_HOME/monitoring_config/LZ/lz_deploy.sh $REGION_CODE || exit 1
fi
#
if [ $DEPLOY_UDAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/UD/ud_deploy.sh $REGION_CODE || exit 1
fi
#
if [ $DEPLOY_YNAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/YN/yn_deploy.sh $REGION_CODE || exit 1
fi
#
if [ $INSTALL_VMAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/VM/vm_install.sh $REGION_CODE || exit 1
fi
#
# if [ $INSTALL_VMFPAGENT == "y" ]
	# then
		# $SCRIPT_HOME/monitoring_config/VM/vm_fp_install.sh $REGION_CODE
# fi
#
if [ $CONFIG_VMAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/VM/vm_configure.sh $REGION_CODE || exit 1
fi
#
if [ $INSTALL_T3AGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/T3/t3_install.sh $REGION_CODE
fi
#
if [ $INSTALL_ISMAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/IS/ism_install.sh $REGION_CODE
fi
#
if [ $DEPLOY_N4AGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/N4/n4_deploy.sh $REGION_CODE || exit 1
fi
#
if [ $DEPLOY_NPAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/NP/np_deploy.sh $REGION_CODE || exit 1
fi
#
if [ $DEPLOY_UMAGENT == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/UM/um_deploy.sh $REGION_CODE
fi
#
#################################################################################################################################################################################
#
if [ $DEPLOY_T6SUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/T6/t6_supports.sh || exit 1
fi
#
if [ $DEPLOY_SKSUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/SK/sk_supports.sh || exit 1
fi
#
if [ $DEPLOY_HTSUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/HT/ht_supports.sh || exit 1
fi
#
if [ $DEPLOY_HTFPSUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/HT/ht_fp_supports.sh || exit 1
fi
#
if [ $DEPLOY_Q7SUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/Q7/q7_supports.sh || exit 1
fi
#
if [ $DEPLOY_MQSUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/MQ/mq_supports.sh || exit 1
fi
#
if [ $DEPLOY_MQFTESUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/MQ/mq_fte_supports.sh || exit 1
fi
#
if [ $DEPLOY_N4SUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/N4/n4_supports.sh || exit 1
fi
#
if [ $DEPLOY_NPSUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/NP/np_supports.sh || exit 1
fi
#
if [ $DEPLOY_D9SUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/D9/d9_supports.sh || exit 1
fi
#
if [ $DEPLOY_P1SUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/P1/p1_supports.sh || exit 1
fi
#
if [ $DEPLOY_LOSUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/LO/lo_supports.sh || exit 1
fi
#
#################################################################################################################################################################################
#
if [ $REBUILD_TEMS == "y" ]
	then
		echo
		echo "======================================================================="
		echo "REBUILDING TEMS"
		echo "======================================================================="
		echo
		/opt/IBM/ITM/bin/itmcmd config -S -t HUB${REGION_CODE} -r -y
		if tac $SCRIPT_HOME/deploy.log | grep -B1000 "REBUILDING TEMS" | grep -q "TEMS configuration completed..."
			then
				echo -e "\nEnd of rebuilding TEMS\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nRebuilding TEMS failed!\n"
				exit 1
		fi
fi
#
if [ $REBUILD_TEPS == "y" ]
	then
		echo
		echo "======================================================================="
		echo "REBUILDING TEPS"
		echo "======================================================================="
		echo
		/opt/IBM/ITM/bin/itmcmd config -A -r -y cq
		if tac $SCRIPT_HOME/deploy.log | grep -B1000 "REBUILDING TEPS" | grep -q "Agent configuration completed..."
			then
				echo -e "\nEnd of rebuilding TEPS\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nRebuilding TEPS failed!\n"
				exit 1
		fi
fi
#
if [ $RESTART_ITM == "y" ]
	then
		echo
		echo "======================================================================="
		echo "RESTARTING ITM"
		echo "======================================================================="
		echo
		service ITMAgents1 stop
		echo -e "\nTurning on SDA..."
		fn=`ls /opt/IBM/ITM/config/ITM${REGION_CODE}_ms_HUB${REGION_CODE}.config`
		sed -e "s/KMS_SDA='N'/KMS_SDA='Y'/g" $fn > sdatmp
		mv -f sdatmp $fn
		sed "s/KMS_SDA=N/KMS_SDA=Y/g" -i /opt/IBM/ITM/config/ms.ini
		#
		echo -e "\nStarting all agents..."
		service ITMAgents1 start
		echo -e "\nService has been restarted! Wait 5 min before ITM components start...\n"
		sleep 300
		#
		run=$(/opt/IBM/ITM/bin/cinfo -r |grep -c "...running")
		if (( ${run} == 12 ))
			then
				echo -e "\nEnd of restarting ITM\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nRestarting ITM failed! "${run}" agent(s) successfully started instead of 12.\n"
				# cq, hd, is, kf, lz, ms, pa, sy, t3, ud, vm, yn
				exit 1
		fi
fi
#
if [ $DEPLOY_K34SUPPORTS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/34/k34_supports.sh $REGION_CODE || exit 1
fi
#
if [ $CONFIG_N4DEVICES == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/N4/n4_dev_config.sh $REGION_CODE
fi
#
#################################################################################################################################################################################
#
if [ $DISABLE_STANDARD_SITUATIONS == "y" ]
	then
		$SCRIPT_HOME/default.scripts/script_allsit_disable.sh
fi
#
if [ $IMPORT_SITUATIONS == "y" ]
	then
		cd $SCRIPT_HOME/default.scripts/SITUATIONS
		./import_sit.sh || exit 1
		cd $SCRIPT_HOME
fi
#
if [ $DISTRIBUTION_SITUATIONS == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/34/k34_sit_distrib.sh			 || exit 1
		$SCRIPT_HOME/monitoring_config/A4/ka4_sit_distrib.sh			 || exit 1
		$SCRIPT_HOME/monitoring_config/CQ/cq_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/HT/ht_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/IS/ism_sit_distrib.sh             || exit 1
		$SCRIPT_HOME/monitoring_config/KR/kr4_sit_distrib.sh             || exit 1
		$SCRIPT_HOME/monitoring_config/LO/lo_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/LZ/lz_sit_distrib.sh 			 || exit 1
		$SCRIPT_HOME/monitoring_config/M6/mqfte_sit_distrib.sh           || exit 1
		$SCRIPT_HOME/monitoring_config/MQ/mq_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/N4/n4_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/NT/nt_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/Q7/q7_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/SK/sk_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/T6/t6_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/UD/ud_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/UM/um_mysql_sit_distrib.sh        || exit 1
		$SCRIPT_HOME/monitoring_config/UM/um_rkasv_sit_distrib.sh        || exit 1
		$SCRIPT_HOME/monitoring_config/UM/um_storwize_sit_distrib.sh     || exit 1
		$SCRIPT_HOME/monitoring_config/UM/um_tsm_sit_distrib.sh    		 || exit 1
		$SCRIPT_HOME/monitoring_config/UM/um_ups_sit_distrib.sh    		 || exit 1
		$SCRIPT_HOME/monitoring_config/UX/ux_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/VM/vm_sit_distrib.sh              || exit 1
		$SCRIPT_HOME/monitoring_config/YN/yn_sit_distrib.sh              || exit 1
fi
#
if [ $ASSOCIATION_SITUATION == "y" ]
	then
		echo
		echo "======================================================================="
		echo "EXPORTING MANAGED SYSTEM ASSIGNMENTS FROM NAVIGATOR VIEW"
		echo "======================================================================="
		echo
		/opt/IBM/ITM/bin/tacmd exportSysAssignments -x $SCRIPT_HOME/SysAssignments.xml -n Physical  -s `hostname` -u sysadmin -p password -f
		if tac $SCRIPT_HOME/deploy.log | grep -B1000 "EXPORTING MANAGED SYSTEM ASSIGNMENTS FROM NAVIGATOR VIEW" | grep "were successfully exported"
			then
				echo -e "\nEnd of exporting managed system assignments\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nExporting managed system assignments failed!"
				exit 1
		fi
		#
		$SCRIPT_HOME/monitoring_config/34/k34_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/A4/ka4_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/CQ/cq_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/HT/ht_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/IS/ism_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/LO/lo_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/LZ/lz_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/M6/mqfte_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/MQ/mq_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/N4/n4_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/NT/nt_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/Q7/q7_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/SK/sk_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/T6/t6_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/UD/ud_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/UM/um_rkasv_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/UM/um_storwize_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/UM/um_tsm_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/UM/um_ups_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/VM/vm_a_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/VM/vm_sit_assoc.sh
		$SCRIPT_HOME/monitoring_config/YN/yn_sit_assoc.sh
fi
#
if [ $CONFIGURE_HISTORY == "y" ]
	then
		$SCRIPT_HOME/monitoring_config/34/k34_history.sh			
		$SCRIPT_HOME/monitoring_config/A4/ka4_history.sh            
		$SCRIPT_HOME/monitoring_config/HT/ht_history.sh             
		$SCRIPT_HOME/monitoring_config/KR/kr2_history.sh            
		$SCRIPT_HOME/monitoring_config/KR/kr4_history.sh            
		$SCRIPT_HOME/monitoring_config/LZ/lz_history.sh             
		$SCRIPT_HOME/monitoring_config/MQ/mq_history.sh             
		$SCRIPT_HOME/monitoring_config/N4/n4_history.sh             
		$SCRIPT_HOME/monitoring_config/NT/nt_history.sh             
		$SCRIPT_HOME/monitoring_config/SK/sk_history.sh             
		$SCRIPT_HOME/monitoring_config/SY/sy_history.sh             
		$SCRIPT_HOME/monitoring_config/UD/ud_history.sh             
		$SCRIPT_HOME/monitoring_config/UM/um_rkasv_history.sh       
		$SCRIPT_HOME/monitoring_config/VM/vm_history.sh             
		$SCRIPT_HOME/monitoring_config/YN/yn_history.sh             
fi
#
if [ $CREATE_OPERATOR_USER == "y" ]
	then
		echo
		echo "======================================================================="
		echo "CREATING NEW USER FOR ITM (OPERATOR)"
		echo "======================================================================="
		echo
		testper=$(/opt/IBM/ITM/bin/tacmd listUsers -u sysadmin -w password | grep operator | cut -f 1 -d ' ')
		if [[ $testper == "operator" ]]
			then
				echo User $testper exists!
			else
				echo User $testper NOT exists and now will be created
				/opt/IBM/ITM/bin/tacmd createUser -i operator -u sysadmin -w password -n operator
				/opt/IBM/ITM/bin/tacmd editUser -i operator -u sysadmin -w password -p "Applications=<All Applications>" "NavigatorViews=Physical, Logical" "MemberOf=*OPERATOR" -f	
				echo -e "tivoli\ntivoli" | passwd operator	
		fi	
		#
		if tac $SCRIPT_HOME/deploy.log | grep -B1000 "CREATING NEW USER FOR ITM (OPERATOR)" | grep -q "all authentication tokens updated successfully"
			then
				echo -e "\nEnd of creating new user for ITM (operator)\n"
				echo -e "-----------------------------------------------------------------------\n"
				sleep $BLOCK_FINISH_PAUSE
			else
				echo -e "\nCreating new user for ITM (operator) failed!\n"
				exit 1
		fi
fi
#
if [ $CUSTOMIZE_WORKSPACES == "y" ]
	then
		$SCRIPT_HOME/default.scripts/customize_workspaces.sh || exit 1
fi
#
if [ $NCO_CHECK == "y" ]
	then
		cp -f $SCRIPT_HOME/default.scripts/NCO_check/nco_check /etc/logrotate.d
		chmod 644 /etc/logrotate.d/nco_check
		cp -f $SCRIPT_HOME/default.scripts/NCO_check/nco_check.sh /etc/cron.hourly
		cp -f $SCRIPT_HOME/default.scripts/NCO_check/restart.sh /StorageDisk/nco_check
		cp -f $SCRIPT_HOME/default.scripts/NCO_check/run.sh /StorageDisk/nco_check
fi
#
if [ $FINAL_REBOOT == "y" ]
	then
		echo
		echo "======================================================================="
		echo "FINAL ADDS AND SERVER REBOOT"
		echo "======================================================================="
		echo
		echo "* soft nofile 10000" >> /etc/security/limits.conf
		echo "* hard nofile 10000" >> /etc/security/limits.conf
		ulimit -n 10000
		useradd pfrroot
		ln -s /opt/IBM/ITM/lx8266/um/scripts/ /root/
		sed "66s@grep -v \${pc}t@grep -v registry\/\${pc}t@" -i /opt/IBM/ITM/bin/uninstall.sh
		#
		echo -e "\nServer is going to reboot now...\n"
		echo -e "\a"
		sleep $BLOCK_FINISH_PAUSE
		reboot
fi
#
warnings_number=$(cat $SCRIPT_HOME/deploy.log | grep -c "WARNING\!\!\!")
echo
echo "======================================================================="
echo "PFR MONITORING DEPLOY SCRIPT COMPLETE"
echo "There are "$warnings_number" warnings!"
echo "======================================================================="
echo
notify-send $a "Installation is complete."
exit 0
