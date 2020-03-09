function install_tf() {
    echo "will use $tf_mode"
    if [ $tf_mode == cpu ]
    then 
        echo "install tensorflow CPU mode"
        pip install tensorflow==1.13.1
    else 
        echo "install tensorflow GPU mode"
        pip install tensorflow-gpu==1.13.1
        conda install cudatoolkit=10.0 -y
        conda install cudnn=7.6 -y
    fi
}

function main() {
    #Make sure gcc is installed 
    conda env remove --name ampligraph || true

    # Create & activate Virtual Environment
    conda create --name ampligraph python=3.6 -y
    source activate ampligraph

    # Install library
    install_tf

    pip install . -v

    # configure dataset location
    export AMPLIGRAPH_DATA_HOME=/var/datasets

    # run flake8 linter
    flake8 ampligraph --max-line-length 120 --ignore=W291,W293 # ignoring some white space related errors

    # run unit tests
    pytest tests

    # build documentation
    cd docs
    make clean autogen html

    # cleanup: remove conda env
    source deactivate
    conda env remove --name ampligraph

}

MEM_THRESHOLD=3000
gpu_candidate=-1

# using GPU = 1, using CPU = 0
tf_mode=gpu

if [ $# -eq 0 ] || [ $1 == "cpu" ]; then
    echo "Test using CPU"
    tf_mode=cpu
    main
fi

if [ $1 == "gpu" ]
then
    total_gpu=$(nvidia-smi -L | grep UUID | wc -l)
    for ((gpu_id=0;gpu_id<$total_gpu;gpu_id++))
    do
        memory="$(nvidia-smi  -q -i $gpu_id -d MEMORY | grep Used)"
        IFS=', ' read -r -a elms <<< ${memory}
        used_memory=${elms[2]}
        
        if [ $used_memory -lt $MEM_THRESHOLD ]
        then 
            gpu_candidate=${gpu_id}
            break
        fi
    done

    if [ $gpu_candidate == -1 ]
    then
        echo "All GPUs are busy. Exit script!"
        exit 1
    else
        tf_mode=gpu
        echo "Will use GPU $gpu_candidate"
        export CUDA_VISIBLE_DEVICES=$gpu_candidate
        main
    fi
fi
