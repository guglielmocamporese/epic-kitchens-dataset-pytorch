#!/bin/bash

# Create annotation folder
mkdir annotations

# Download EK-55 annotations
git clone https://github.com/epic-kitchens/epic-kitchens-55-annotations.git annotations/epic-kitchens-55-annotations

# Download EK-100 annotations
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git annotations/epic-kitchens-100-annotations

# Download RULSTM annotations
git clone https://github.com/fpv-iplab/rulstm.git annotations/rulstm
