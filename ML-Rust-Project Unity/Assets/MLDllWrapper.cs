using System.Runtime.InteropServices;

public static class MlDllWrapper
{
    
    //////////////////////////////////////////////////LINEAR MODEL///////////////////////////////////////////////////////////
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "create_linear_model")]
    public static extern System.IntPtr CreateLinearModel(int inputSize);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_linear_model")]
    public static extern double PredictLinearModel(System.IntPtr model,
        double[] inputs, int inputSize, bool isClassification);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "train_linear_model_class")]
    public static extern void trainLinearClass(System.IntPtr model, double[] inputs, int inputSize, int inputSampleSize,
        double[] ouputs, int outputSize, int outputSampleSize, double learningRate, bool isClassification, int epochs);
    

    //////////////////////////////////////////////////MLP MODEL///////////////////////////////////////////////////////////
    
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "create_mlp_model")]
    public static extern System.IntPtr CreateMLPModel(int numberLayer, int[] neuronesCount);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_mlp_model")]
    public static extern System.IntPtr PredictMLPModel(System.IntPtr model,
        double[] inputs, int inputSize, int numberLayer, int[] npl, bool isClassification);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "train_mlp_model_class")]
    public static extern void trainMLPModelClass(System.IntPtr model, int numberLayer, int datasetSize, int[] npl, double[] input, int inputSize, int inputSampleSize,
        double[] ouputs, int outputSize, int outputSampleSize, int epochs, double learningRate, bool isClassification);
    
    
    //////////////////////////////////////////////////RBF///////////////////////////////////////////////////////////

    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "init_RBF")]
    public static extern System.IntPtr InitRBF(double[] input, int inputSize, int inputSampleSize,
        double[] ouputs, int outputSize, int outputSampleSize, int k);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_RBF")]
    public static extern System.IntPtr PredictRBF(System.IntPtr rbf, double[] inputX, int inputSize);


    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "delete_linear_model")]
    public static extern void DeleteLinearModel(System.IntPtr model);
}