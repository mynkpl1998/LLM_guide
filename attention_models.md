# Attention Models

## Neural Machine Translation (NMT)

NMT is a task where the machine needs to translate from one human language to another.
It uses encoder and a decoder architecture to translate from one language to another.

Example: English-to-German

### NMT using LSTM

NMT using LSTM/GRU are done using a Seq2Seq model. 
* These models map a variable length input sequence to a fixed-length memory using an encoder.
* The encoded output is then fed to a decoder which can generates the output.
* The models can have different length of inputs and outputs which is often required for machine translation.
* LSTM/GRU are used to avoid vanishing and exploding gradients problem for long sequences.

![Alt text](content/nmt_lstms.png)

#### Seq2Seq Encoder

The word of the sentance is called as *token*. The token are converted to a fixed size vector using an embedding layer. The embedded input along with hidden state is passed to the LSTM cell to generate a new hidden state which encodes the information from previous cell.

![Alt text](content/lstm_encoder.png)

#### Seq2Seq Decoder

The final hidden state of the encoder is passed to the decoder with specialized start of sequence *SOS* token to kick of the translation.

![Alt text](content/lstm_decoder.png)


#### Information bottleneck

The main issue with Seq2Seq models is the fixed size memory. For longer sequences, the fixed size memory is enough to pass the whole information from encoder to decoder.

![Alt text](content/information_bottleneck.png)

As the sequence size increases, the fixed size memory is not able to compress/retain the information in longer sequences leading to poor model performance.

#### Information bottleneck - Solution 1

Instead of using the final/compressed hidden state, use all the hidden states in the decoder.
However, this would require huge amount of memory for long sequences.

![Alt text](content/bottleneck_sol1.png)

#### Information bottleneck - Solution 2 - Attention

We can add an attention layer which can learn to attend which hidden state is most important in the sequence.

![Alt text](content/generated_sol2.png)

#### Performance of Attention Seq2Seq models

Given enough fixed size memory, the models with attention outperformed the existing non attention based models.

![Alt text](content/attn_seq2seq_perf.png)

#### How to use all hidden states ?

It is now well established that, passing all the hidden state to decoder instead of compressing all the hidden state into single hidden state is more usefull. But the question remains, how to do it efficiently without blowing up the memory and compute requirements.

![Alt text](content/hiddn_states.png)
