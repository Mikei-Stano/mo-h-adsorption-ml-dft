#!/bin/bash
# Helper script to find and copy VASP POTCARs
# Run this to locate POTCARs on your HPC system

echo "=========================================="
echo "VASP POTCAR Finder & Copier"
echo "=========================================="

# Try common locations
echo -e "\n🔍 Searching for VASP POTCARs..."

POTCAR_LOCATIONS=(
    "/opt/vasp/potcar"
    "/usr/local/vasp/potcar"
    "/home/$USER/.vasp/potcar"
    "/apps/vasp/potcar"
    "/soft/vasp/potcar"
    "$VASP_PP_PATH"
    "$VASP_PSCTR"
)

# Also check module-loaded VASP paths
if command -v module &> /dev/null; then
    echo "Checking module-loaded VASP..."
    module load vasp 2>/dev/null
    VASP_BIN=$(which vasp 2>/dev/null)
    if [ ! -z "$VASP_BIN" ]; then
        VASP_DIR=$(dirname $(dirname "$VASP_BIN"))
        POTCAR_LOCATIONS+=("$VASP_DIR/potcar")
        POTCAR_LOCATIONS+=("$VASP_DIR/lib")
    fi
fi

echo -e "\nCommon POTCAR locations to check:"
for loc in "${POTCAR_LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "  ✓ FOUND: $loc"
        ls -la "$loc" | head -5
        POTCAR_DIR="$loc"
        break
    else
        echo "  ✗ Not found: $loc"
    fi
done

if [ -z "$POTCAR_DIR" ]; then
    echo -e "\n❌ Could not find POTCAR directory!"
    echo -e "\n📋 NEXT STEPS:"
    echo "1. Ask your HPC administrator: 'Where are VASP POTCARs installed?'"
    echo "2. Set the path manually:"
    echo "   export VASP_POTCAR_DIR='/actual/path/to/potcar'"
    echo "3. Run this script again"
    exit 1
fi

echo -e "\n✓ Using POTCAR directory: $POTCAR_DIR"

# Show available element POTCARs
echo -e "\n📦 Available elements:"
ls -d "$POTCAR_DIR"/*/ 2>/dev/null | xargs -n1 basename | head -20

echo -e "\n=========================================="
echo "STEP 1: Manual Setup Instructions"
echo "=========================================="
echo -e "\n⚠️  IMPORTANT: You must handle POTCARs carefully!"
echo -e "\nFor MoS2 surfaces:"
echo "  cd VASP_inputs/MoS2_\(100\)/"
echo "  cat $POTCAR_DIR/Mo/POTCAR $POTCAR_DIR/S/POTCAR > POTCAR"
echo -e "\nFor MoSe2 surfaces:"
echo "  cd VASP_inputs/MoSe2_\(100\)/"
echo "  cat $POTCAR_DIR/Mo/POTCAR $POTCAR_DIR/Se/POTCAR > POTCAR"
echo -e "\nFor MoP surfaces:"
echo "  cd VASP_inputs/MoP_\(100\)/"
echo "  cat $POTCAR_DIR/Mo/POTCAR $POTCAR_DIR/P/POTCAR > POTCAR"
echo -e "\nFor Mo2N surfaces:"
echo "  cd VASP_inputs/Mo2N_\(100\)/"
echo "  cat $POTCAR_DIR/Mo/POTCAR $POTCAR_DIR/N/POTCAR > POTCAR"

echo -e "\n=========================================="
echo "STEP 2: Auto-copy (if POTCAR found)"
echo "=========================================="

if [ ! -z "$POTCAR_DIR" ] && [ -f "$POTCAR_DIR/Mo/POTCAR" ]; then
    echo "Copying POTCARs to all directories..."
    
    # MoS2
    for dir in VASP_inputs/MoS2_*/; do
        if [ -d "$dir" ]; then
            cat "$POTCAR_DIR/Mo/POTCAR" "$POTCAR_DIR/S/POTCAR" > "$dir/POTCAR" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✓ $(basename $dir)"
            fi
        fi
    done
    
    # MoSe2
    for dir in VASP_inputs/MoSe2_*/; do
        if [ -d "$dir" ] && [ -f "$POTCAR_DIR/Se/POTCAR" ]; then
            cat "$POTCAR_DIR/Mo/POTCAR" "$POTCAR_DIR/Se/POTCAR" > "$dir/POTCAR" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✓ $(basename $dir)"
            fi
        fi
    done
    
    # MoP
    for dir in VASP_inputs/MoP_*/; do
        if [ -d "$dir" ] && [ -f "$POTCAR_DIR/P/POTCAR" ]; then
            cat "$POTCAR_DIR/Mo/POTCAR" "$POTCAR_DIR/P/POTCAR" > "$dir/POTCAR" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✓ $(basename $dir)"
            fi
        fi
    done
    
    # Mo2N
    for dir in VASP_inputs/Mo2N_*/; do
        if [ -d "$dir" ] && [ -f "$POTCAR_DIR/N/POTCAR" ]; then
            cat "$POTCAR_DIR/Mo/POTCAR" "$POTCAR_DIR/N/POTCAR" > "$dir/POTCAR" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ✓ $(basename $dir)"
            fi
        fi
    done
    
    echo -e "\n✓ POTCARs copied successfully!"
    
    # Verify
    echo -e "\nVerifying..."
    for dir in VASP_inputs/*/; do
        if [ -f "$dir/POTCAR" ]; then
            size=$(stat -f%z "$dir/POTCAR" 2>/dev/null || stat -c%s "$dir/POTCAR" 2>/dev/null)
            echo "  ✓ $(basename $dir): $(($size / 1024)) KB"
        else
            echo "  ✗ $(basename $dir): MISSING"
        fi
    done
else
    echo "❌ Could not find POTCAR files automatically"
    echo "   Please follow manual instructions above"
fi

echo -e "\n=========================================="
echo "NEXT: Submit jobs"
echo "=========================================="
echo "sbatch VASP_inputs/submit.sh"
echo "or"
echo "for dir in VASP_inputs/*/; do cd \"\$dir\" && sbatch submit.sh && cd ../..; done"
