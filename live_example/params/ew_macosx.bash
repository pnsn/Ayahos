# Create an Earthworm environment on MacOSX?!
# This file should be sourced by a bourne shell wanting 
# to run or build an EARTHWORM system under Linux.

# Set environment variables describing your Earthworm directory/version
# export EW_HOME=/opt/earthworm/
export EW_HOME=/opt/earthworm
export PULSE_HOME=/Users/nates/Code/PULSE
export EW_VERSION=earthworm_7.10
export SYS_NAME=`hostname`

# Set environment variables used by earthworm modules at run-time
# Path names must end with the slash "/"
export EW_INSTALLATION=INST_UW
export EW_PARAMS=${PULSE_HOME}/live_example/params/
export EW_LOG=${PULSE_HOME}/live_example/log/
export EW_DATA_DIR=${PULSE_HOME}/live_example/data/

#
# Database (Oracle) related environment
#

# not used in linux build
#export SCHEMA_DIR=${EW_HOME}/${EW_VERSION}/src/oracle/schema-working
#export APPS_DIR=${EW_HOME}/${EW_VERSION}/src/oracle/apps
#export WEB_DOC_DIR=${EW_HOME}/${EW_VERSION}/WEB_DOC/DBMS/API-DOC

#
# Web server related environment
#
export WEB_DIR=${EW_HOME}/web

# Tack the earthworm bin directory in front of the current path
# Also add Oracle paths to the current path.
#set path=( ${EW_HOME}/${EW_VERSION}/bin /opt/oracle/bin /var/opt/oracle $path )
export PATH=${EW_HOME}/$EW_VERSION/bin:$PATH

# Set up library path for dynamically loaded libraries:
#export OPENWINHOME   /usr/openwin
#export ORACLE_HOME   /opt/oracle
#export COMPILER_DIR /opt/SUNWspro
#export LD_LIBRARY_PATH="${OPENWINHOME}/lib:${ORACLE_HOME}/lib:${COMPILER_DIR}/lib:/usr/lib"

# Set environment variables for compiling earthworm modules
#export GLOBALFLAGS="-Dlinux -D__i386 -D_LINUX -D_INTEL -D_USE_SCHED  -D_USE_PTHREADS -D_USE_TERMIOS -I${EW_HOME}/${EW_VERSION}/include"
export GLOBALFLAGS="-D_MACOSX -D_INTEL -D_USE_PTHREADS -D_USE_SCHED -I${EW_HOME}/${EW_VERSION}/include"

export CFLAGS=$GLOBALFLAGS
export KEEP_STATE=""
