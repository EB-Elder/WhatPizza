mod rbf;
mod utils;

use std::slice::{from_raw_parts};
use std::os::raw::c_char;
use std::ffi::CString;
use rand::Rng;
use nalgebra::*;
use crate::rbf::RBF;




//////////////////////////////////////////////////LINEAR MODEL///////////////////////////////////////////////////////////

#[no_mangle]
pub extern fn create_linear_model(input_size: usize) -> *mut Vec<f64> {

    //Initialisation du model lineaire avec une taille défini + un biais
    let mut weights = Vec::with_capacity(input_size + 1);

    //Initialisation des poids
    for i in 0..input_size {
        weights.push(rand::thread_rng().gen_range(-1.0, 1.0));

    }
    //Initialisation du poids du biais
    weights.push(rand::thread_rng().gen_range(-1.0, 1.0));

    //Fuite mémoire volontaire afin de pouvoir renvoyer un pointeur
    let boxed_weights = Box::new(weights);
    let boxed_ref = Box::leak(boxed_weights);

    boxed_ref
}

#[no_mangle]
pub extern fn predict_linear_model(model: *mut Vec<f64>, inputs: *mut f64, inputs_size: usize,  is_classification: bool) -> f64
{
    let boxed_model;
    let inputs_slice;

    unsafe {
        //Récupération des contenus des pointeurs
        boxed_model = model.as_ref().unwrap();
        inputs_slice = from_raw_parts(inputs, inputs_size);
    }

    let mut sum = 0.0;

    //prédictions
    for i  in 0..inputs_slice.len() {
        sum += inputs_slice[i] * boxed_model[i];
    }

    //Sachant que le biais = 1.0
    sum += boxed_model[inputs_slice.len()];

    if is_classification
    {
        return utils::sign(sum);
    }
    sum


}

#[no_mangle]
pub extern fn train_linear_model_class(model: *mut Vec<f64>, inputs: *mut f64, input_size: usize, input_sample_size: usize,
                                       output: *mut f64, output_size: usize, output_sample_size: usize, learning_rate: f64,  is_classification: bool, epochs: i32)
{
    let mut boxed_model;
    let mut input_slice;
    let mut output_slice;
    unsafe {
        boxed_model = &mut *model;
        input_slice = from_raw_parts(inputs, input_size);
        output_slice = from_raw_parts(output, output_size);
    }

    let dataset_size = output_size/output_sample_size;


    for it in 0..epochs
    {
        if is_classification
        {
            let biais = 1.0;

            let mut k = rand::thread_rng().gen_range(0, dataset_size);
            let mut sampled_input: Vec<f64> = Vec::new();
            let mut sampled_output: Vec<f64> = Vec::new();

            for i in input_sample_size * k..input_sample_size * (k + 1)
            {
                sampled_input.push(input_slice[i]);
            }

            sampled_input.push(1.0);

            for i in output_sample_size * k..output_sample_size * (k + 1)
            {
                sampled_output.push(output_slice[i]);
            }

            let result = utils::_predict_linear_model(model, &sampled_input, input_size, is_classification);

            //Mise a jour des poids
            // W[i] = W[i] + r * error * Input[i]

            for l in 0..input_sample_size + 1
            {
                for z in 0..output_sample_size
                {
                    boxed_model[l] = boxed_model[l] + learning_rate * (sampled_output[z] - result) * sampled_input[l];
                }
            }

        } else {
            let X: DMatrix<f64> = DMatrix::<f64>::from_column_slice(input_size / input_sample_size, input_sample_size, input_slice);
            let Y: DMatrix<f64> = DMatrix::<f64>::from_column_slice(output_size / output_sample_size, output_sample_size, output_slice);


            let mut W: DMatrix<f64> = ((X.transpose() * &X).try_inverse().unwrap().clone());


            W = (W * X.transpose()) * &Y;

            for i in 0..W.len()
            {
                boxed_model[i] = W.get(i).unwrap().clone();
            }
        }
    }


}

//////////////////////////////////////////////////MLP MODEL///////////////////////////////////////////////////////////

#[no_mangle]
pub extern fn create_mlp_model(number_layer: usize, neurones_count: *mut i32) -> *mut Vec<f64> {


    //Initialisation du model lineaire avec une taille défini + un biais

    //Nombre de neurones par couche cachées
    let mut npl;
    unsafe {
        npl = from_raw_parts(neurones_count, number_layer);
    }

    let mut weights = vec![0.0; 0];

    //Ajout de toutes les neurones et des biais

    //Initialisation des poids
    for l in 0..(npl.len())
    {
        if l==0
        {
            // pour la couche d'entrée 100.0 étant pour nous l'équivalent de NONE
            weights.push(100.0);
            continue
        }
        for i in 0..(npl[l-1]+1)
        {
            for j in 0..(npl[l] + 1)
            {
                weights.push(rand::thread_rng().gen_range(-1.0, 1.0))
            }
        }
    }

    //Fuite mémoire volontaire afin de pouvoir renvoyer un pointeur
    let boxed_weights = Box::new(weights);
    let boxed_ref = Box::leak(boxed_weights);

    boxed_ref

}

#[no_mangle]
pub extern fn predict_mlp_model(model: *mut Vec<f64>,
                                                  inputs: *mut f64, inputs_size: usize, number_layer: usize, neurones_count: *mut i32,  is_classification: bool) -> *mut c_char
{
    let boxed_model;
    let inputs_slice;
    let neurones_count_slice;
    let L = number_layer - 1;


    unsafe {
        //Récupération des contenus des pointeurs
        boxed_model = &mut *model;
        inputs_slice = from_raw_parts(inputs, inputs_size);
        neurones_count_slice = from_raw_parts(neurones_count, number_layer);
    }

    let mut vec_boxed_model = utils::weight_array_1dto3d(boxed_model, neurones_count_slice);

    let mut neurones_values = vec![vec![0.0; 0]; number_layer];

    neurones_values[0].push(1.0);
    for i in 0..inputs_size
    {
        neurones_values[0].push(inputs_slice[i])
    }


    for i in 1..neurones_count_slice.len(){

        neurones_values[i].push(1.0);
        for j in 0..neurones_count_slice[i]
        {
            neurones_values[i].push(0.0)
        }
    }

    for l in 1..(L + 1)
    {
        for j in 1..(neurones_count_slice[l] + 1)
        {
            let mut sum:f64 = 0.0;
            for i in 0..(neurones_count_slice[l - 1] + 1)
            {
                sum += neurones_values[l - 1][i as usize] * vec_boxed_model[l][i as usize][j as usize];
            }
            if l == L && !is_classification {
                neurones_values[l][j as usize] = sum;
            }
            else {
                neurones_values[l][j as usize] = sum.tanh();
            }
        }
    }


    let mut result_string = "".to_string();

    for i in 1..neurones_values[L].len()
    {
        let tmp = neurones_values[L][i].to_string()+";";
        result_string.push_str(&tmp);
    }

    let pntr = CString::new(result_string).unwrap().into_raw();

    pntr



}

#[no_mangle]
pub extern fn train_mlp_model_class(model: *mut Vec<f64>, number_layer: usize, dataset_size: usize, neurones_count: *mut i32,  inputs: *mut f64, input_size: usize, input_sample_size: usize,
                                output: *mut f64, output_size: usize, output_sample_size: usize, epochs: i32, learning_rate: f64,  is_classification: bool)
{
    let mut boxed_model;
    let input_slice;
    let output_slice;
    let neurones_count_slice;
    let L = number_layer - 1;




    unsafe {
        boxed_model = &mut *model;
        input_slice = from_raw_parts(inputs, input_size);
        output_slice = from_raw_parts(output, output_size);
        neurones_count_slice = from_raw_parts(neurones_count, number_layer);
    }

    let mut vec_boxed_model = utils::weight_array_1dto3d(boxed_model, neurones_count_slice);

    let mut deltas:Vec<Vec<f64>> = Vec::new();

    let mut neurones_values = vec![vec![0.0; 0]; number_layer];

    neurones_values[0].push(1.0);
    for i in 0..input_sample_size
    {
        neurones_values[0].push(0.0);
    }

    for i in 1..neurones_count_slice.len(){

        neurones_values[i].push(1.0);
        for j in 0..neurones_count_slice[i]
        {
            neurones_values[i].push(0.0)
        }
    }

    for i in 0..number_layer
    {
        let mut tmp:Vec<f64> = Vec::new();
        for j in 0..neurones_count_slice[i]+1
        {
            if j == 0{
                tmp.push(1.0);
            }
            else {
                tmp.push(0.0);
            }

        }
        deltas.push(tmp);
    }

    for it in 0..epochs
    {


        let mut k = rand::thread_rng().gen_range(0, dataset_size);
        let mut sampled_input:Vec<f64> = Vec::new();
        let mut sampled_output:Vec<f64> = Vec::new();

        for i in input_sample_size * k..input_sample_size * (k + 1)
        {
            sampled_input.push(input_slice[i]);
        }

        for j in 0..sampled_input.len()
        {
            neurones_values[0][j+1] = sampled_input[j];
        }

        for i in output_sample_size * k..output_sample_size * (k + 1)
        {
            sampled_output.push(output_slice[i]);
        }




        ///////////////////////PREDICTIONS///////////////////////////////////////
        for l in 1..(L + 1)
        {
            for j in 1..(neurones_count_slice[l] + 1)
            {
                let mut sum:f64 = 0.0;
                for i in 0..(neurones_count_slice[l - 1] + 1)
                {
                    sum += neurones_values[l - 1][i as usize] * vec_boxed_model[l][i as usize][j as usize];
                }
                if l == L && !is_classification {
                    neurones_values[l][j as usize] = sum;
                }
                else {
                    neurones_values[l][j as usize] = sum.tanh();
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////

        for j in 1..neurones_count_slice[L] + 1
        {
            deltas[L][j as usize] = neurones_values[L][j as usize] - sampled_output[j as usize - 1];
            if is_classification
            {
                deltas[L][j as usize] = deltas[L][j as usize] * (1.0 - utils::power(neurones_values[L][j as usize], 2));
            }
        }

        for l in (2..L + 1).rev()
        {
            for i in 0..neurones_count_slice[l - 1]+1
            {
                let mut sum = 0.0;
                for j in 1..neurones_count_slice[l] + 1
                {
                    sum += vec_boxed_model[l][i as usize][j as usize] * deltas[l][j as usize]
                }
                deltas[l as usize - 1][i as usize] = (1.0 - utils::power(neurones_values[l as usize - 1][i as usize], 2)) * sum;
            }
        }

        for l in 1..L + 1
        {
            for i in 0..neurones_count_slice[l - 1] + 1
            {
                for j in 1..neurones_count_slice[l] + 1
                {
                    vec_boxed_model[l][i as usize][j as usize] -= learning_rate * neurones_values[l - 1][i as usize] * deltas[l][j as usize];
                }
            }
        }
    }

    utils::weight_array_3dto1d(boxed_model, &vec_boxed_model, &neurones_count_slice);

}

////////////////////////////////////////////////////////RBF////////////////////////////////////////////////////////////////////////////
#[no_mangle]
pub extern fn init_RBF(inputs: *mut f64, input_size: usize, input_sample_size: usize,
                       output: *mut f64, output_size: usize, output_sample_size: usize, k: i32,) //->*mut RBF
{
    let mut input_vec;
    let mut trainning_output;

    unsafe {
        input_vec = from_raw_parts(inputs, input_size).to_vec();
        trainning_output = from_raw_parts(output, output_size).to_vec();
    }

    let trainning_input = utils::_1dto2dVec(&input_vec, input_size/input_sample_size, input_sample_size);

    let mut test_input: Vec<Vec<f64>> = Vec::new();
    let mut test_output: Vec<f64> = Vec::new();

    for i in (trainning_input.len()*0.8 as usize)..trainning_input.len()
    {
        test_input.push(trainning_input[i].clone());
    }

    for i in (trainning_output.len()*0.8 as usize)..trainning_output.len()
    {
        test_output.push(trainning_output[i]);
    }

    let mut RBF = RBF::init(trainning_input, trainning_output, test_input, test_output, output_sample_size as i32, k);

    RBF.fit();

    /*let boxed_RBF = Box::new(RBF);
    let boxed_ref = Box::leak(boxed_RBF);

    boxed_ref*/
}

#[no_mangle]
pub extern fn predict_RBF(RBF_model: *mut RBF, input_X: *mut f64, input_size: i32) -> i32
{
    let mut tst;
    let mut boxed_ref;
    unsafe {
        tst = Box::from_raw(RBF_model);
        boxed_ref = Box::leak(tst);
    }

    let resulted_index = boxed_ref.predict(input_X, input_size as usize);

    resulted_index

}

#[no_mangle]
pub extern fn delete_linear_model(model: *mut Vec<f64>)
{
    unsafe {
        let boxed_model = Box::from_raw(model);
    };
}

