H adsorption calculation: MoS2 on (111) surface

Steps:
1. Copy POTCAR from VASP: cp $VASP_POTCAR_DIR/POTCAR .
2. Run: mpirun -np 16 vasp > vasp.out
3. Extract energy from OUTCAR
