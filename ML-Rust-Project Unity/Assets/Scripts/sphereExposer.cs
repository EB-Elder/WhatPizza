using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class sphereExposer : MonoBehaviour
{
    
    [SerializeField] public Transform myTransform;
    [SerializeField] private MeshRenderer _meshRenderer;

    public void ChangeMaterial(Material newMat)
    {
        _meshRenderer.material = newMat;
    }

    public void changeZ(float Z)
    {
        myTransform.position = new Vector3(myTransform.position.x, myTransform.position.y, Z);
    }
}
