using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

public class TestDll : MonoBehaviour
{
    [SerializeField]
    private sphereExposer[] spheres;

    [SerializeField] private Material blueMat;
    [SerializeField] private Material redMat;
    [SerializeField] private Material greenMat;

    [SerializeField] private int epochs;
    
    [Space(10)]
    
    [Header("Input Value in float")]
    
    [SerializeField] private int[] npl = new[] {4, 4};
    [SerializeField] private double[] inputs = new[] {1.0, 1.0}; 
    private int numberLayer;
    
    
    [Range(0.0f, 1.0f)] 
    [SerializeField] private double learningRate = 0.8f;

    [SerializeField] private bool isClassification = true;
    
    double[] trainningInput = new []
    {

         0.34068145, -0.26114152,
         0.33633634,  0.72782065,
        -0.07562696, -0.1684039 ,
         0.60135366,  0.52758022,
         0.35229312,  0.79349299,
         0.37481457, -0.51672255,
         0.27822503, -0.91802473,
        -0.72829057,  0.50048368,
         0.84380973,  0.6277717 ,
        -0.54367566, -0.20464249,
         0.04454513,  0.12764608,
        -0.93682818,  0.26244532,
         0.323177  , -0.02986667,
         0.15819571,  0.81723335,
        -0.77995191, -0.3743708 ,
        -0.73899397, -0.48255471,
         0.68676538,  0.00422727,
        -0.74474725, -0.30412313,
        -0.62192887,  0.4988258 ,
        -0.82853251, -0.85484962,
         0.04869682, -0.13792705,
        -0.41714507, -0.93878862,
         0.25942264, -0.43846772,
         0.62403706, -0.18733077,
        -0.87755936, -0.90658985,
         0.61110068, -0.20819132,
        -0.79877158,  0.50170128,
         0.05962299,  0.78260946,
        -0.15639158,  0.04238416,
        -0.68390984,  0.95275266,
        -0.58016413,  0.77962843,
         0.44636434,  0.07209964,
         0.42994503,  0.14730415,
        -0.82492263, -0.12163574,
        -0.12175932, -0.95186268,
         0.14789679,  0.69186296,
        -0.26175257,  0.93858828,
        -0.05618285, -0.58789736,
        -0.9747978 ,  0.60544274,
        -0.47061584,  0.10210318,
        -0.07624899, -0.54677267,
         0.74909011,  0.35968964,
         0.84754295, -0.24714253,
         0.44661855, -0.87028549,
        -0.75381015, -0.68673835,
         0.06579795,  0.03714897,
         0.81982488,  0.45432225,
         0.73100943, -0.53660882,
         0.4595062 , -0.3966522 ,
         0.33407243,  0.31284609
        
    };

    double[] trainninOuput = new []
    {
        -1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,
         1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,
         1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0
    };

    // Start is called before the first frame update

    private int modelSize = 2;
    private System.IntPtr MyModel;
    List<double[]> testDataSet = new List<double[]>();

    private List<System.IntPtr> _myMulticlassModel = new List<IntPtr>();

    private List<double> _multiclassResult = new List<double>();
    List<double[]> _outputTrainningMulticlass = new List<double[]>();
    
    private int _inputSize;
    private int _outputSize;
    
    void randomizeSpheres()
    {
        foreach (sphereExposer sphere in spheres)
        {
            sphere.myTransform.position = new Vector3(Random.Range(-1.0f, 1.0f), Random.Range(-1.0f, 1.0f), 0.0f);
        }
    }

    void CreateLinearMulticlass()
    {
        for (int i = 0; i < _outputSize; i++)
        {
            _myMulticlassModel.Add(MlDllWrapper.CreateLinearModel(_inputSize));
        }
    }

    void predictLinearMulticlass()
    {
        
        
        testDataSet = new List<double[]>();
        
        foreach (sphereExposer sphere in spheres)
        {
            Vector3 pos = sphere.myTransform.position;
            double[] tmp = new[] {(double)pos.x,  pos.y};
            testDataSet.Add(tmp);
        }      
        
        for (int j = 0; j < testDataSet.Count; j++)
        {
            
            _multiclassResult.Clear();

            int bestModel = 0;
            double bestValue = 0.0f;
            for (int i = 0; i < _outputSize; i++)
            {
                double currentValue = MlDllWrapper.PredictLinearModel(_myMulticlassModel[i], testDataSet[j], _inputSize, isClassification);

                print($"Model numero {i} = {currentValue}");
                if (currentValue > bestValue)
                {
                    bestModel = i;
                }
                
            }

            switch (bestModel)
            {
                default:
                    spheres[j].ChangeMaterial(blueMat);
                    break;
                case 1:
                    spheres[j].ChangeMaterial(redMat);
                    break;
                case 2:
                    spheres[j].ChangeMaterial(greenMat);
                    break;
            }

        }
        
        
        
    }

    void trainLinearMulticlass()
    {
        
        print("Starting trainning");
        _outputTrainningMulticlass.Clear();
        
        _outputTrainningMulticlass = new List<double[]>();

        int outputLength = trainninOuput.Length / _outputSize;
        
        for (int i = 0; i < _outputSize; i++)
        {
            List<double> tmp = new List<double>();
            for (int j = 0; j < outputLength; j++)
            {
                tmp.Add(trainninOuput[j * _outputSize + i]);
            }
            _outputTrainningMulticlass.Add(tmp.ToArray());
        }
        

        for (int i = 0; i < _outputTrainningMulticlass.Count; i++)
        {
            MlDllWrapper.trainLinearClass(_myMulticlassModel[i], trainningInput, trainningInput.Length, _inputSize, _outputTrainningMulticlass[i],
                    _outputTrainningMulticlass[i].Length, 1, learningRate, isClassification, epochs);
        }
        
        print("Trainning Finished");
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
            var result = MlDllWrapper.PredictLinearModel(MyModel, testDataSet[i], testDataSet[i].Length, true);
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
            1.77144984, 1.65116022,
       1.6399382 , 1.59086106,
       1.13172281, 1.07656568,
       1.25930466, 1.15616839,
       1.62027458, 1.02434234,
       1.20974377, 1.67894966,
       1.14747175, 1.68923731,
       1.18610008, 1.33349034,
       1.17019652, 1.36966102,
       1.77313756, 1.34577527,
       1.21880784, 1.14429813,
       1.57213441, 1.86181584,
       1.26238583, 1.02049052,
       1.36135625, 1.74683675,
       1.25246501, 1.72803973,
       1.59040201, 1.09210728,
       1.03503882, 1.32942747,
       1.45489275, 1.20543673,
       1.84288301, 1.23780841,
       1.38633379, 1.47091256,
       1.72083348, 1.33304295,
       1.85868546, 1.60458102,
       1.88602417, 1.42306061,
       1.32915664, 1.86049581,
       1.88070804, 1.28109471,
       1.20123114, 1.84995572,
       1.16960876, 1.40463481,
       1.26327802, 1.04502057,
       1.470317  , 1.7205709 ,
       1.22694328, 1.17398128,
       1.20439113, 1.57437446,
       1.20823462, 1.41741227,
       1.52589379, 1.19055702,
       1.79973395, 1.85894671,
       1.64965302, 1.07957112,
       1.65437772, 1.32799941,
       1.29988881, 1.8935194 ,
       1.33803235, 1.40598044,
       1.32354564, 1.58565161,
       1.26749372, 1.89174172,
       1.42161561, 1.83953343,
       1.3276691 , 1.48204737,
       1.60477696, 1.77871508,
       1.62998447, 1.01766031,
       1.79551964, 1.49571658,
       1.81173498, 1.57695283,
       1.26163979, 1.5023208 ,
       1.08627909, 1.04747032,
       1.04980366, 1.38939534,
       1.2219717 , 1.27705727,
       2.23605306, 2.59413503,
       2.77685403, 2.00404757,
       2.0066633 , 2.4098892 ,
       2.54427068, 2.17796376,
       2.35643478, 2.0340604 ,
       2.84798073, 2.08387991,
       2.40811825, 2.53029238,
       2.48724392, 2.70784423,
       2.89330259, 2.46002952,
       2.00747218, 2.28250935,
       2.76663243, 2.73110769,
       2.62935116, 2.16883541,
       2.45924454, 2.47625261,
       2.20241569, 2.84985256,
       2.44886553, 2.34179714,
       2.82035084, 2.76644604,
       2.00624709, 2.68028073,
       2.10996734, 2.30257441,
       2.61867637, 2.09334176,
       2.81107613, 2.62011081,
       2.61263338, 2.34433106,
       2.39186818, 2.03076351,
       2.00517557, 2.84927318,
       2.88016631, 2.05953701,
       2.76529191, 2.13036204,
       2.44159905, 2.34056267,
       2.83242774, 2.13010271,
       2.03230813, 2.03263851,
       2.49300779, 2.59158412,
       2.62589154, 2.31362233,
       2.40615634, 2.68615363,
       2.58995807, 2.31942761,
       2.29743199, 2.77681675,
       2.62920349, 2.27121495,
       2.73512514, 2.39542262,
       2.11741817, 2.0995095 ,
       2.26296159, 2.80198306,
       2.7259719 , 2.27534142,
       2.68590626, 2.50868839,
       2.65301566, 2.07738944,
       2.58668408, 2.50266752,
       2.06830937, 2.53841298,
       2.01687053, 2.23185386,
       2.50158022, 2.34561901,
       2.03185534, 2.13053901,
       2.33521746, 2.2553881 ,
       2.70786891, 2.57408026,
       2.54775324, 2.5371683 ,
       2.15909133, 2.5675018 ,
       2.79557181, 2.77711779
        };
        
        double[] trainningOutput = new[] {    1.0,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
             1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f};

        MlDllWrapper.trainLinearClass(MyModel, trainningInput, trainningInput.Length, 2,
                trainningOutput, trainningOutput.Length, 1, learningRate, true, epochs);
        print("Model finished train");
 
    }

    void trainMLPModel(double[] trainningInput, double[] trainningOutput, int nplSize)
    {
        print("Model start trainning");

        print(npl[0]);
        
        MlDllWrapper.trainMLPModelClass(MyModel, nplSize, 4, npl, trainningInput, trainningInput.Length, npl[0],
                trainningOutput, trainningOutput.Length,  _outputSize, epochs, learningRate, true);
        
        print("Model finished train");
    }
    
    void Start()
    {
        _inputSize = npl[0];
        _outputSize = npl.Last();
        numberLayer = npl.Length;
        //MyModel = MlDllWrapper.CreateLinearModel(modelSize);
        CreateLinearMulticlass();
        randomizeSpheres();
        //trainModel();
        
        //predictOnDataSet();

        //MyModel = MlDllWrapper.CreateMLPModel(numberLayer, npl);
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.A))
        {
            /*List<double[]> test = new List<double[]>();

            test.Add(new []{0.0, 1.0});
            test.Add(new []{1.0, 0.0});
            test.Add(new []{0.0, 0.0});
            test.Add(new []{1.0, 1.0});

            for (int i = 0; i < 4; i++)
            {
                print(MlDllWrapper.PredictMLPModel(MyModel, test[i], inputSize, numberLayer, npl, true));
            }*/
            
            
            /*double result = MlDllWrapper.PredictMLPModelClassification(MyModel, inputs, inputSize, numberLayer, npl);
            if (result > 0)
            {
                print(1);
            }
            else
            {
                print(-1);
            }*/
            //predictOnDataSet();
            predictLinearMulticlass();
        }
        
        if (Input.GetKeyDown(KeyCode.T))
        {
            //trainMLPModel(trainningInput, trainninOuput, numberLayer);
            //trainModel();
            trainLinearMulticlass();
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            MlDllWrapper.DeleteLinearModel(MyModel);
        }
    }
}
