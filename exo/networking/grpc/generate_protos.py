#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from grpc_tools import protoc

def generate_protos():
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent.parent.parent
    proto_dir = root_dir / "exo" / "networking" / "grpc"
    
    # Create __init__.py if it doesn't exist
    init_file = proto_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
    
    # Generate both node_service and file_service protos
    proto_files = ["node_service.proto", "file_service.proto"]
    
    for proto_file in proto_files:
        protoc.main([
            "grpc_tools.protoc",
            f"--proto_path={root_dir}",
            f"--python_out={root_dir}",
            f"--grpc_python_out={root_dir}",
            f"exo/networking/grpc/{proto_file}"
        ])
        
        print(f"Generated GRPC code for {proto_file}")

if __name__ == "__main__":
    generate_protos()