# Create an Earthworm environment on ARM Linux
# This file should be sourced by a Bourne shell wanting
# to run or build an EARTHWORM system under Linux on ARM.

# CAUTION: ARM EW should be considered experimental until there is enough
# experience to be sure the code behaves correctly using PC compilers and
# ARM compilers.  For example, the C standard leaves the signedness of the
# char type up to the compiler.  On a PC, the gcc char type is signed; on
# ARM, the gcc char type is unsigned.  In addition, there may be operand
# alignment requirements that are not correct and will not be found until
# the module with the problem code is run.

# For running EW on Linux, if any ports are used, make sure that your
# selected ports for wave_serverV or exports are below the range specified
# by your kernel for dynamic port allocation 
# (see sysctl net.ipv4.ip_local_port_range)

# Set environment variables describing your Earthworm directory/version


# Use value from elsewhere IF defined (eg from .bashrc)
# otherwise use the value after the :-
# export EW_HOME="${EW_INSTALL_HOME:-/opt/earthworm}"
# export EW_VERSION="${EW_INSTALL_VERSION:-earthworm_7.10}"
# EW_RUN_DIR="${EW_RUN_DIR:-${EW_HOME}/run_working}"
# Or set your own values directly
export EW_HOME=/usr/local/earthworm
export EW_VERSION=earthworm_7.10
EW_RUN_DIR=$EW_HOME/run_working

# This next env var is required if you run statmgr:
export SYS_NAME=`hostname`

# Set environment variables used by Earthworm modules at run-time
# Path names must end with the slash "/"
export EW_INSTALLATION="${EW_INSTALL_INSTALLATION:-INST_UNKNOWN}"
export EW_PARAMS="${EW_RUN_DIR}/params/"
export EW_LOG="${EW_RUN_DIR}/log/"
export EW_DATA_DIR="${EW_RUN_DIR}/data/"

# Tack the Earthworm bin directory in front of the current path
export PATH="${EW_HOME}/${EW_VERSION}/bin:${PATH}"

# Set environment variables for compiling Earthworm modules

# Be explicit about which compilers to use (only CC is checked for validity)
export CC=gcc
export CXX=g++

if [ "${CC}" = "gcc" ] ; then

   # Set EW_BITS=64 to build for 64-bit target (note that with EW_BITS=64
   # size of 'long' type changes from 4 bytes to 8 bytes)

   # gcc target prefix is "arm" on 32-bit ARM, "aarch64" on 64-bit ARM
   CC_BITS=`${CC} -v 2>&1 |
            awk '/^Target:/{
                    split( $2, arch, "-" );
                    if ( arch[1] == "aarch64" )
                       print( "64" )
                    else
                       print( "32" )
                 }'`
   export EW_BITS="${CC_BITS}"

   # Warning flags for compiler:
   WARNFLAGS="-Wall -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unknown-pragmas -Wno-pragmas -Wformat -Wdeclaration-after-statement"
   # Extra flags for enabling more warnings during code development:
   #WARNFLAGS="-Wall -Wextra -Wcast-align -Wpointer-arith -Wbad-function-cast -Winline -Wundef -Wnested-externs -Wshadow -Wfloat-equal -Wno-unused-parameter -Wformat -Wdeclaration-after-statement"

   # -D_FILE_OFFSET_BITS=64 is for Large Filesystem Support (64-bit offsets) on a 32-bit compile; it is benign on a 64-bit compile

   # gcc -pthread compiles and links with POSIX Threads support

   # Set initial defaults for the gmake implicit .c.o and .cpp.o target rules

   # C compiler flags (also used for ld flags)
   export CFLAGS="-g -pthread ${WARNFLAGS}"
   # C++ compiler flags
   export CXXFLAGS="${CFLAGS}"
   # C preprocessor defs and includes
   export CPPFLAGS="-D_LINUX -Dlinux -D_INTEL -D_USE_SCHED -D_USE_PTHREADS -D_USE_TERMIOS -D_FILE_OFFSET_BITS=64 -I${EW_HOME}/${EW_VERSION}/include"

   # Earthworm makefiles override CFLAGS=$(GLOBALFLAGS) (for now), and not all
   # makefiles have been converted yet to use the implicit gmake target rules
   export GLOBALFLAGS="${CFLAGS} ${CPPFLAGS} EWBITS=64 -m64 -fPIC"

   # Use the Linux makefile.unix options
   export PLATFORM="LINUX"

else

   echo "CC must be gcc"

fi

# Pick a Fortran compiler
# gfortran (freeware, part of GNU GCC)
export FC=gfortran

if [ "${FC}" = "gfortran" ] ; then

   # Extra run-time checks: -fcheck=all
   #export FFLAGS="-g -O -Wuninitialized -Werror=line-truncation -ffpe-trap=invalid,zero,overflow -fcheck=all,no-array-temps -fbacktrace"
   export FFLAGS="-g -O -Wuninitialized -Werror=line-truncation -ffpe-trap=invalid,zero,overflow -fbacktrace"

   # FC_MAIN_IS_C is not needed with gfortran
   #export FC_MAIN_IS_C=-nofor-main

else

   echo "FC must be gfortran"

fi

# Alternatively, you can hard-code values here:
#export FC='...'
#export FFLAGS='...'
