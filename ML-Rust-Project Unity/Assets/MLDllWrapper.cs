using System.Runtime.InteropServices;

public static class MlDllWrapper
{
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "my_add")]
    public static extern double MyAdd(double a, double b);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "create_linear_model")]
    public static extern System.IntPtr CreateLinearModel(int inputSize);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "create_mlp_model")]
    public static extern System.IntPtr CreateMLPModel(int numberHiddenLayer, int[] neuronesCount);

    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_linear_model_classification")]
    public static extern double PredictLinearModelClassification(System.IntPtr model,
        double[] inputs, int inputSize);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_mlp_model_classification")]
    public static extern double PredictMLPModelClassification(System.IntPtr model,
        double[] inputs, int inputSize, int numberLayer, int[] npl);
    
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "get_weights")]
    public static extern double getWeights(System.IntPtr model, int index);
    
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "predict_linear_model_multiclass_classification")]
    public static extern System.IntPtr PredictLinearModelMultiClassClassification(System.IntPtr model,
        double[] inputs, int inputSize, int classCount);

    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "train_model_class")]
    public static extern void trainModelClass(System.IntPtr model, double[] inputs, int inputSize, int inputSampleSize,
                                                double[] ouputs, int outputSize, int outputSampleSize, double learningRate);
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "train_mlp_model_class")]
    public static extern void trainMLPModelClass(System.IntPtr model, int numberHiddenLayer, int[] npl, double[] input, int inputSize, int inputSampleSize,
        double[] ouputs, int outputSize, int epochs, double learningRate);
    
    
    
    [DllImport("_2021_5A_3DJV_RustMLDll", EntryPoint = "delete_linear_model")]
    public static extern void DeleteLinearModel(System.IntPtr model);
}