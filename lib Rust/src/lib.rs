use std::slice::{from_raw_parts};
use rand::Rng;

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
pub extern fn create_mlp_model(input_size: usize, number_hidden_layer: usize, neurones_count: *mut i32) -> *mut Vec<f64> {


    //TODO: CREATION DE TABLEAU 2D
    //Initialisation du model lineaire avec une taille défini + un biais

    let mut neurones_count_slice;
    let mut final_capacity:usize = input_size +  1 + number_hidden_layer;
    unsafe {
        neurones_count_slice = from_raw_parts(neurones_count, number_hidden_layer);
    }


    for i in 0..number_hidden_layer{
        final_capacity+=neurones_count_slice[i] as usize;
    }

    //Ajout de toutes les neurones et des biais
    let mut weights = Vec::with_capacity(final_capacity);
    //Initialisation des poids
    for i in 0..final_capacity {
        weights.push(rand::thread_rng().gen_range(-1.0, 1.0));

    }


    //Fuite mémoire volontaire afin de pouvoir renvoyer un pointeur
    let boxed_weights = Box::new(weights);
    let boxed_ref = Box::leak(boxed_weights);

    boxed_ref

}

#[no_mangle]
pub extern fn get_weights(model: *mut Vec<f64>, index: usize) -> f64
{
    let boxed_model;
    unsafe {
        boxed_model = model.as_ref().unwrap();
    }
    boxed_model[index]
}

#[no_mangle]
pub extern fn predict_mlp_model_classification(model: *mut Vec<f64>,
                                                  inputs: *mut f64, inputs_size: usize, number_hidden_layer: usize, neurones_count: *mut i32) -> f64 {
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

    let biais = 1.0;

    sum += biais * boxed_model[inputs_slice.len()];

    if sum >= 0.0
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }


}


#[no_mangle]
pub extern fn predict_linear_model_classification(model: *mut Vec<f64>,
                                                  inputs: *mut f64, inputs_size: usize) -> f64 {
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

        let biais = 1.0;

        sum += biais * boxed_model[inputs_slice.len()];

        if sum >= 0.0
        {
            return 1.0;
        }
        else
        {
            return -1.0;
        }


}


#[no_mangle]
pub extern fn predict_linear_model_multiclass_classification(model: *mut Vec<f64>,
                                                             inputs: *mut f64, inputs_size: usize, class_count: usize) -> *mut f64 {
    let boxed_model;
    let inputs_slice;

    unsafe {
        boxed_model = model.as_ref().unwrap();
        inputs_slice = from_raw_parts(inputs, inputs_size);
    }

    let mut sum = 0.0;

    // TODO
    for elt in inputs_slice {
        sum += elt;
    }

    let mut result = Vec::with_capacity(class_count);
    result.push(42.0);
    result.push(51.0);
    result.push(69.0);

    let boxed_result = result.into_boxed_slice();
    let boxed_result_leaked = Box::leak(boxed_result);
    return boxed_result_leaked.as_mut_ptr();
}


#[no_mangle]
pub extern fn train_mlp_model_class(model: *mut Vec<f64>, number_hidden_layer: usize, neurones_count: *mut i32,  inputs: *mut f64, input_size: usize, input_sample_size: usize,
                                output: *mut f64, output_size: usize, output_sample_size: usize, learning_rate: f64)
{
    let mut boxed_model;
    let mut input_slice;
    let mut output_slice;
    let mut neurones_count_slice;
    let mut neurones_values:Vec<Vec<f64>> = Vec::with_capacity(number_hidden_layer);

    unsafe {
        boxed_model = &mut *model;
        input_slice = from_raw_parts(inputs, input_size);
        output_slice = from_raw_parts(output, output_size);
        neurones_count_slice = from_raw_parts(neurones_count, number_hidden_layer);
    }

    for i in 0..neurones_count_slice.len(){
        let mut tmp= Vec::with_capacity(i);
        for k in 0..i{
            tmp.push(0.0);
        }
        neurones_values.push(tmp);
    }

    let dataset_size = output_size/output_sample_size;


    let mut result=0.0;
    let biais = 1.0;

    for i in 0..dataset_size{



        for j in (i * output_sample_size).. (i * output_sample_size + output_sample_size)
        {

            //TODO: Optimiser la prédictions ici
            //Prédictions
            result=0.0;

            for hidden_layer in 0..number_hidden_layer
            {
                if number_hidden_layer == 0
                {
                    for neurones in input_slice.len()
                    {
                        neurones_values[hidden_layer][neurones] = 0.0;
                    }
                }
            }

            for k in (i * input_sample_size).. (i * input_sample_size + input_sample_size)
            {

            }
        }
    }

}


#[no_mangle]
pub extern fn train_model_class(model: *mut Vec<f64>, inputs: *mut f64, input_size: usize, input_sample_size: usize,
                                output: *mut f64, output_size: usize, output_sample_size: usize, learning_rate: f64)
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


    let mut result=0.0;
    let biais = 1.0;

    for i in 0..dataset_size{


        for j in (i * output_sample_size).. (i * output_sample_size + output_sample_size){

            //TODO: Optimiser la prédictions ici
            //Prédictions
            result=0.0;
            for k in (i * input_sample_size).. (i * input_sample_size + input_sample_size){

                result += input_slice[k]*boxed_model[k%input_sample_size];

            }
            result += biais*boxed_model[input_sample_size];

            if result >= 0.0
            {
                result = 1.0;
            }
            else { result = -1.0; }


            //Mise a jour des poids
            // W[i] = W[i] + r * error * Input[i]
            if result != output_slice[j]
            {
                let mut last_index = 0;
                for l in (i * input_sample_size).. (i * input_sample_size + input_sample_size)
                {
                    boxed_model[l%input_sample_size] = boxed_model[l%input_sample_size] + learning_rate * (output_slice[j] - result) * input_slice[l];
                    last_index = l;
                }
                boxed_model[(last_index%input_sample_size) + 1] = boxed_model[(last_index%input_sample_size) + 1] + learning_rate * (output_slice[j] - result);

            }

        }
    }

}

#[no_mangle]
pub extern fn delete_native_array(arr: *mut f64) {
    unsafe {
        Box::from_raw(arr);
    }
}


#[no_mangle]
pub extern fn delete_linear_model(model: *mut Vec<f64>) {
    unsafe {
        let boxed_model = Box::from_raw(model);
    };
}


#[no_mangle]
pub extern fn my_add(a: f64, b: f64) -> f64 {
    a + b
}