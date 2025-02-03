# Disclaimer!!! 
* Tokenizer hasn't been implemented yet so each "token" is character level 
* Because of this lyrics don't make sense and is only producing a combination of random words. 

# Model Architecture
- 5 Transformer Encoder blocks
- Each block includes 
    * Positional + "Token" Embeddings (Each token is character level)
    * First Layer Norm followed by Masked MultiHeadedSelfAttention
    * Second Layer Norm followed by a Feed Forward Net
    * There is also a residual stream

# Hyperparameters
- context_length = 200
- n_emb = 360
- n_heads = 6
- dropout = 0.2
- head_size = n_emb // n_heads
- n_layers = 5
- epochs = 2000
- batch_size = 32
- learning_rate = 1e-3
- epoch_intervals = 500

# Sample Outputs
- Smaller Model - 1000 characters (without tokenization): 
    fifter under memore time fall down lefinish me and palk that i guessure though all the ungry ain t and with the decords 
    poor don t surf chair penish me sterpeseder goh come and everyna and getsinglass keeping like me bed like my feetist caling 
    chair some and i got so home and every don t come and i don t get all need i thought a gen instand em proud setting his got 
    hing in the crash of feel nyer  say put in your kinda cupsetter soon glass stop out listen the dim tryna mood no no none girl 
    call my ni**as you can chails git this plank that book stage the shit sying me to the sunshine s getting up watch smee but is 
    you hhest i was so way the red mistare and i m tryin to try but i look first tim starped far i adin out for the yursends in the 
    middle and no it hard the oir the rains don t got time get every simpty this is is mind to mistartendentry we stop all mage and we ll 
    singing never be rong we sturn you lie checkin they sks three depotain my ridon out of the skisted why i could you kno

- Bigger Model with Full Dataset - 1000 characters (without tokenization):
    so got up in your blue another her wanna see my song i just like you don t mind it have to go to we don t have to want to hold my 
    hand thousand i ll take it a phone a garded about my heard and passin old six place with the been reason in the radio and vayams 
    long help clearives and kollin caro mixs but kept in help lookin cup at on all shockin with a honey milkshit didn t hunder but all 
    my pretty legs trip drip just after phone c re out my really guns around it s finger102wardress hungry ay fade a but had a party but 
    now a sea ya setic the the is i kinda so ainter runs the runnin run run run friends savide king friends me everybody s sounds that was 
    the rain your grever great died see my suffores daddy suffore this man colarshiny everydes die down in s man i m just lie up gave a down 
    stop livin i gues it when you lave a guess the can inters in the wallers sipty but glass lately 3232been sittin overight and it front in 
    a guitatious somethin it walk out tous my comes comes could i wish