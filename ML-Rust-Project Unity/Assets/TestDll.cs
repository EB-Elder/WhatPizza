using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TestDll : MonoBehaviour
{
    [SerializeField]
    private sphereExposer[] spheres;

    [SerializeField] private Material blueMat;
    [SerializeField] private Material redMat;

    [SerializeField] private int epochs;
    
    // Start is called before the first frame update

    private int modelSize = 2;
    private System.IntPtr MyModel;
    List<double[]> testDataSet = new List<double[]>();
    
    void randomizeSpheres()
    {
        foreach (sphereExposer sphere in spheres)
        {
            sphere.myTransform.position = new Vector3(Random.Range(0.0f, 5.0f), Random.Range(0.0f, 5.0f), 0.0f);
        }
    }

    void predictOnDataSet()
    {
        testDataSet = new List<double[]>();
        
        foreach (sphereExposer sphere in spheres)
        {
            Vector3 pos = sphere.myTransform.position;
            double[] tmp = new[] {(double)pos.x,  pos.y};
            testDataSet.Add(tmp);
        }        
        
        for (int i = 0; i < testDataSet.Count; i++)
        {
            var result = MlDllWrapper.PredictLinearModelClassification(MyModel, testDataSet[i], testDataSet[i].Length);
            print("X : " + testDataSet[i][0] + "Y : " + testDataSet[i][1] + " : " + result);
            if (result > 0)
                spheres[i].ChangeMaterial(redMat);
            else
                spheres[i].ChangeMaterial(blueMat);
                
        }
    }

    void trainModel()
    {
        print("Model start trainning");

        double[] trainningInput = new[]
        {
            1.0, 1.0,
            2.0, 3.0,
            3.0, 3.0
        };
        
        double[] trainningOutput = new[] {   1.0,
                                            -1.0,
                                            -1.0};

        for (int i = 0; i < epochs; i++)
        {
            MlDllWrapper.trainModelClass(MyModel, trainningInput, trainningInput.Length, 2,
                trainningOutput, trainningOutput.Length, 1, 0.5);
        }
        print("Model finished train");
        
        
    }
    
    void Start()
    {
        MyModel = MlDllWrapper.CreateLinearModel(modelSize);
        randomizeSpheres();
        //trainModel();


        
        predictOnDataSet();
    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetKeyDown(KeyCode.A))
        {
            predictOnDataSet();
        }
        
        if (Input.GetKeyDown(KeyCode.T))
        {
            trainModel();
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            MlDllWrapper.DeleteLinearModel(MyModel);
        }
    }
}
