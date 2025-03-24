import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    Inizializza i pesi del layer in base al numero di input (fan_in)
    usando un intervallo uniforme: (-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """
    fan_in = layer.weight.data.size()[1]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)

class PortfolioActor(nn.Module):
    """
    Rete neurale per la policy (Actor) adattata per portafogli multi-asset.

    Input:
      - state: vettore di stato che include feature di ogni asset, posizioni attuali e metriche di portafoglio.

    Output:
      - Azioni: vettore di azioni, una per ogni asset nel portafoglio.
    """
    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=128, fc3_units=64, use_batch_norm=True):
        """
        Inizializza la rete dell'Actor.
        
        Parametri:
        - state_size: dimensione dello stato (feature di tutti gli asset + posizioni + metriche)
        - action_size: numero di asset nel portafoglio (un'azione per asset)
        - seed: seme per riproducibilità
        - fc1_units, fc2_units, fc3_units: dimensioni dei layer nascosti
        - use_batch_norm: se utilizzare la batch normalization
        """
        super(PortfolioActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batch_norm = use_batch_norm
        
        # Layer lineari
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
        # Batch normalization (opzionale)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.bn3 = nn.BatchNorm1d(fc3_units)
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.bias.data.fill_(0)
        # Il layer finale viene inizializzato con valori più piccoli
        self.fc4.weight.data.uniform_(-3e-4, 3e-4)
        self.fc4.bias.data.fill_(0)
    
    def forward(self, state):
        """
        Forward pass dell'Actor avanzato.
        """
        batch_size = state.size(0)
        features_per_asset = (state.size(1) - (state.size(1) % self.action_size)) // self.action_size
        
        # Codifica gli asset
        encoded_state = self.asset_encoder(state, self.action_size)
        
        # Applica attenzione se abilitata
        if self.use_attention:
            encoded_state = self.apply_attention(encoded_state, batch_size)
        
        # Debug della dimensione
        #print(f"DEBUG - encoded_state shape dopo attenzione: {encoded_state.shape}, fc1 weight shape: {self.fc1.weight.shape}")
        
        # Adatta dinamicamente il layer FC1 se necessario
        if self.fc1.weight.shape[1] != encoded_state.size(1):
            print(f"ATTENZIONE: Ridimensionamento del layer FC1 da {self.fc1.weight.shape[1]} a {encoded_state.size(1)}")
            old_fc1_out_features = self.fc1.weight.shape[0]
            self.fc1 = torch.nn.Linear(encoded_state.size(1), old_fc1_out_features)
            # Reinizializza i pesi
            fan_in = self.fc1.weight.data.size()[1]
            lim = 1.0 / np.sqrt(fan_in)
            self.fc1.weight.data.uniform_(-lim, lim)
            self.fc1.bias.data.fill_(0)
        
        # Feed-forward con batch norm
        x = F.relu(self.bn1(self.fc1(encoded_state)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Output layer
        return self.fc3(x)
    
class PortfolioCritic(nn.Module):
    """
    Rete neurale per il Critic adattata per portafogli multi-asset.
    
    Input:
      - state: vettore di stato che include feature di ogni asset, posizioni e metriche
      - action: vettore di azioni, una per ogni asset
      
    Output:
      - Valore Q: singolo valore che rappresenta il valore stimato dell'azione
    """
    def __init__(self, state_size, action_size, seed=0, fcs1_units=256, fc2_units=128, fc3_units=64, use_batch_norm=True):
        """
        Inizializza la rete del Critic.
        
        Parametri:
        - state_size: dimensione dello stato
        - action_size: numero di asset nel portafoglio
        - seed: seme per riproducibilità
        - fcs1_units, fc2_units, fc3_units: dimensioni dei layer nascosti
        - use_batch_norm: se utilizzare la batch normalization
        """
        super(PortfolioCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_batch_norm = use_batch_norm
        
        # Layer per elaborare lo stato e l'azione
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        
        # Batch normalization (opzionale)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(fcs1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.bn3 = nn.BatchNorm1d(fc3_units)
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.bias.data.fill_(0)
        # Il layer finale viene inizializzato con valori più piccoli
        self.fc4.weight.data.uniform_(-3e-4, 3e-4)
        self.fc4.bias.data.fill_(0)
    
    def forward(self, state, action):
        """
        Esegue il forward pass del Critic.
        
        Args:
            state (Tensor): vettore di stato
            action (Tensor): vettore di azioni
            
        Returns:
            Tensor: il valore Q della coppia (stato, azioni)
        """
        # Concatena stato e azione
        x = torch.cat((state, action), dim=1)
        
        # Applica i layer con attivazioni e batch norm (se abilitata)
        if self.use_batch_norm:
            x = F.relu(self.bn1(self.fcs1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
        else:
            x = F.relu(self.fcs1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        
        # Layer di output (singolo valore Q)
        return self.fc4(x)

class AssetEncoder(nn.Module):
    """
    Modulo ottimizzato per codificare le feature di ciascun asset in modo indipendente.
    Con gestione robusta delle dimensioni e debugging integrato.
    """
    def __init__(self, features_per_asset, encoding_size=16, seed=0, output_size=None):
        """
        Inizializza il codificatore di asset.
        
        Parametri:
        - features_per_asset: numero di feature per singolo asset
        - encoding_size: dimensione suggerita dell'encoding per asset (verrà adattata se necessario)
        - seed: seme per riproducibilità
        - output_size: dimensione di output opzionale (diversa da encoding_size)
        """
        super(AssetEncoder, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.features_per_asset = features_per_asset
        self.encoding_size = encoding_size
        self.output_size = output_size or encoding_size  # Usa output_size se specificato
        
        # Layer di encoding
        self.fc1 = nn.Linear(features_per_asset, 32)
        self.fc2 = nn.Linear(32, self.output_size)
        
        # Debug info
        self.first_forward_done = False
        
        # Inizializza i parametri
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
    
    def forward(self, state, num_assets):
        """
        Codifica le feature di ciascun asset indipendentemente, con diagnostica e
        gestione robusta delle dimensioni.
        
        Args:
            state (Tensor): stato completo [batch_size, features_per_asset*num_assets + extra]
            num_assets: numero di asset nel portafoglio
            
        Returns:
            Tensor: feature codificate [batch_size, num_assets*encoding_size + extra]
        """
        batch_size = state.size(0)
        total_features = state.size(1)
        
        # Calcola quante feature dovrebbero essere usate per gli asset
        asset_features_total = num_assets * self.features_per_asset
        
        # Stampa info diagnostiche solo al primo passaggio
        if not self.first_forward_done:
            print(f"AssetEncoder - Primo forward pass")
            print(f"  state shape: {state.shape}")
            print(f"  num_assets: {num_assets}")
            print(f"  features_per_asset: {self.features_per_asset}")
            print(f"  asset_features_total: {asset_features_total}")
            print(f"  total_features: {total_features}")
            self.first_forward_done = True
        
        # Verifica se ci sono abbastanza feature nello stato
        if total_features < asset_features_total:
            print(f"ATTENZIONE: Lo stato ha {total_features} feature, ma ci aspettiamo almeno {asset_features_total}")
            # Gestione di emergenza: paddiamo lo stato
            padding_needed = asset_features_total - total_features
            padding = torch.zeros(batch_size, padding_needed, device=state.device)
            padded_state = torch.cat([state, padding], dim=1)
            state = padded_state
            total_features = state.size(1)
        
        # Estrai le feature di ciascun asset
        asset_features = state[:, :asset_features_total]
        
        # Verifica se ci sono feature extra da preservare
        extra_features = None
        if total_features > asset_features_total:
            extra_features = state[:, asset_features_total:]
        
        # Riorganizza per processare ogni asset indipendentemente
        try:
            asset_features = asset_features.view(batch_size, num_assets, self.features_per_asset)
        except RuntimeError as e:
            print(f"Errore nel reshape dell'encoder: {e}")
            print(f"Dimensioni: batch_size={batch_size}, num_assets={num_assets}, features_per_asset={self.features_per_asset}")
            print(f"asset_features.shape={asset_features.shape}, asset_features.numel()={asset_features.numel()}")
            print(f"Prodotto atteso: {batch_size * num_assets * self.features_per_asset}")
            
            # Fallback di emergenza
            if asset_features.numel() == 0:
                # Crea un tensor di zeri di dimensioni appropriate
                print("ATTENZIONE: asset_features vuoto, creazione di emergenza")
                asset_features = torch.zeros(batch_size, num_assets, self.features_per_asset, device=state.device)
            else:
                # Ridimensionamento sicuro
                total_elements = asset_features.numel()
                safe_features_per_asset = total_elements // (batch_size * num_assets)
                print(f"Tentativo di recupero con safe_features_per_asset={safe_features_per_asset}")
                
                if safe_features_per_asset <= 0:
                    # Situazione critica, crea un tensor di emergenza
                    asset_features = torch.zeros(batch_size, num_assets, self.features_per_asset, device=state.device)
                else:
                    # Adatta la dimensione delle feature per asset
                    self.features_per_asset = safe_features_per_asset
                    asset_features = asset_features[:, :batch_size * num_assets * safe_features_per_asset]
                    asset_features = asset_features.view(batch_size, num_assets, safe_features_per_asset)
        
        # Codifica ogni asset
        x = F.relu(self.fc1(asset_features))
        asset_encodings = F.relu(self.fc2(x))
        
        # Appiattisci gli encoding
        asset_encodings = asset_encodings.view(batch_size, num_assets * self.output_size)
        
        # Ricombina con le feature extra se presenti
        if extra_features is not None:
            return torch.cat((asset_encodings, extra_features), dim=1)
        
        return asset_encodings

class EnhancedPortfolioActor(nn.Module):
    """
    Versione avanzata dell'Actor che utilizza un encoder per asset e meccanismi di attenzione.
    Con gestione dinamica delle dimensioni e fix per batch size=1.
    """
    def __init__(self, state_size, action_size, features_per_asset, seed=0, 
             fc1_units=256, fc2_units=128, encoding_size=32, use_attention=True,
             attention_size=None, encoder_output_size=None):
        """
        Inizializza l'Actor avanzato con gestione automatica delle dimensioni.
        
        Parametri:
        - state_size: dimensione dello stato completo
        - action_size: numero di asset nel portafoglio
        - features_per_asset: feature per singolo asset
        - encoding_size: dimensione suggerita dell'encoding per asset (verrà adattata)
        - attention_size: dimensione suggerita per il layer di attenzione (verrà adattata)
        - encoder_output_size: dimensione di output suggerita per l'encoder (verrà adattata)
        """
        super(EnhancedPortfolioActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.features_per_asset = features_per_asset
        self.use_attention = use_attention
        
        # Encoder per asset - la dimensione di output verrà adattata automaticamente
        self.asset_encoder = AssetEncoder(
            features_per_asset, 
            encoding_size=encoding_size, 
            seed=seed,
            output_size=encoder_output_size
        )
        
        # Extra feature (posizioni attuali + metriche di portfolio)
        self.extra_features = state_size - (features_per_asset * action_size)
        
        # Non inizializziamo qui i layer di attenzione o FC
        # Li creeremo dinamicamente nel primo forward pass
        self.attention = None
        self.value = None
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        
        # Sostituiamo BatchNorm con LayerNorm che funziona anche con batch_size=1
        self.ln1 = None
        self.ln2 = None
        
        # Salva le dimensioni per i layer FC
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        
        # Flag per tracciare se i layer sono già stati inizializzati
        self.layers_initialized = False
        
        # Dimensioni dinamiche da determinare durante il primo forward pass
        self.effective_encoding_size = None
        self.fc_input_size = None
    
    def initialize_layers(self, encoded_size):
        """
        Inizializza dinamicamente tutti i layer necessari una volta che conosciamo
        le dimensioni effettive dell'encoding.
        """
        print(f"Inizializzazione dinamica dei layer. Input FC size: {encoded_size}")
        
        # Input layer
        self.fc1 = nn.Linear(encoded_size, self.fc1_units)
        
        # Usiamo LayerNorm invece di BatchNorm per supportare batch_size=1
        self.ln1 = nn.LayerNorm(self.fc1_units)
        
        # Hidden layer
        self.fc2 = nn.Linear(self.fc1_units, self.fc2_units)
        self.ln2 = nn.LayerNorm(self.fc2_units)
        
        # Output layer
        self.fc3 = nn.Linear(self.fc2_units, self.action_size)
        
        # Inizializza i pesi
        self.reset_parameters()
        
        # Imposta il flag di inizializzazione
        self.layers_initialized = True
    
    def reset_parameters(self):
        """
        Inizializza i pesi dei layer con valori appropriati.
        """
        # Inizializza solo se i layer esistono
        if self.fc1 is not None:
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc1.bias.data.fill_(0)
        
        if self.fc2 is not None:
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc2.bias.data.fill_(0)
        
        if self.fc3 is not None:
            self.fc3.weight.data.uniform_(-3e-4, 3e-4)
            self.fc3.bias.data.fill_(0)
    
    def initialize_attention_layers(self, effective_encoding_size):
        """
        Inizializza i layer di attenzione con la dimensione effettiva.
        """
        print(f"Inizializzazione dei layer di attenzione con dimensione: {effective_encoding_size}")
        
        # Crea il layer di attenzione per calcolare i pesi
        self.attention = nn.Linear(effective_encoding_size, 1)
        
        # Crea il layer value che trasforma gli asset
        self.value = nn.Linear(effective_encoding_size, effective_encoding_size)
        
        # Inizializza con valori appropriati
        fan_in = effective_encoding_size
        lim = 1.0 / np.sqrt(fan_in)
        self.attention.weight.data.uniform_(-lim, lim)
        self.attention.bias.data.fill_(0)
        self.value.weight.data.uniform_(-lim, lim)
        self.value.bias.data.fill_(0)
        
        # Salva la dimensione effettiva per riferimento
        self.effective_encoding_size = effective_encoding_size
    
    def apply_attention(self, encoded_assets, batch_size):
        """
        Applica meccanismo di attenzione tra asset con gestione robusta delle dimensioni.
        """
        # Controlla le dimensioni effettive
        total_size = encoded_assets.size(1)
        
        # Verifica se abbiamo feature extra alla fine dello stato
        extra_features = None
        if self.extra_features > 0 and total_size > self.action_size * self.features_per_asset:
            # Estrai le feature extra (ultime self.extra_features feature dello stato)
            extra_features = encoded_assets[:, -self.extra_features:]
            # Rimuovi le feature extra per il calcolo dell'attenzione
            encoded_assets = encoded_assets[:, :-self.extra_features]
            # Aggiorna total_size
            total_size = encoded_assets.size(1)
        
        # Calcola quante feature dovrebbe avere ogni asset
        if total_size % self.action_size == 0:
            # Caso ideale: la dimensione è esattamente divisibile
            effective_encoding_size = total_size // self.action_size
        else:
            # Caso non ideale: dobbiamo aggiungere padding
            effective_encoding_size = (total_size + self.action_size - 1) // self.action_size
            padding_needed = effective_encoding_size * self.action_size - total_size
            
            if padding_needed > 0:
                padding = torch.zeros(batch_size, padding_needed, device=encoded_assets.device)
                encoded_assets = torch.cat([encoded_assets, padding], dim=1)
                total_size = encoded_assets.size(1)
                #print(f"Aggiunto padding: {padding_needed}. Nuova dimensione: {total_size}")
        
        # Verifica se i layer di attenzione esistono e hanno la dimensione corretta
        if self.attention is None or self.value is None or self.effective_encoding_size != effective_encoding_size:
            self.initialize_attention_layers(effective_encoding_size)
        
        # Reshape per processare ogni asset separatamente [batch, num_assets, features_per_asset]
        try:
            assets = encoded_assets.view(batch_size, self.action_size, effective_encoding_size)
        except RuntimeError as e:
            print(f"Errore nel reshape: {e}")
            print(f"Dimensioni: encoded_assets={encoded_assets.shape}, batch_size={batch_size}, action_size={self.action_size}, effective_encoding_size={effective_encoding_size}")
            print(f"Prodotto: {batch_size * self.action_size * effective_encoding_size}, Elementi disponibili: {encoded_assets.numel()}")
            
            # Fallback di emergenza: ri-adatta le dimensioni
            total_elements = encoded_assets.numel()
            safe_encoding_size = total_elements // (batch_size * self.action_size)
            print(f"Tentativo di recupero con encoding_size={safe_encoding_size}")
            
            # Aggiungi padding se necessario
            if batch_size * self.action_size * safe_encoding_size > total_elements:
                print("ERRORE: calcolo dimensioni non valido")
                # Ritorna un tensore vuoto come fallback
                return encoded_assets
            
            # Ricrea il layer di attenzione
            self.initialize_attention_layers(safe_encoding_size)
            
            # Riprova il reshape
            assets = encoded_assets.view(batch_size, self.action_size, safe_encoding_size)
            effective_encoding_size = safe_encoding_size
        
        # Calcola punteggi di attenzione
        attention_scores = self.attention(assets).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        
        # Trasforma gli asset
        values = self.value(assets)
        
        # Applica attenzione
        context = (attention_weights * values).sum(dim=1)
        
        # Espandi il contesto e concatena con gli encoding originali
        context_expanded = context.unsqueeze(1).expand(-1, self.action_size, -1)
        enhanced_assets = torch.cat((assets, context_expanded), dim=2)
        
        # Appiattisci il risultato
        enhanced_size = enhanced_assets.size(2) * self.action_size
        flattened = enhanced_assets.view(batch_size, enhanced_size)
        
        # Ricombina con le feature extra se presenti
        if extra_features is not None:
            flattened = torch.cat((flattened, extra_features), dim=1)
        
        return flattened
    
    def forward(self, state):
        """
        Forward pass dell'Actor avanzato con inizializzazione dinamica dei layer.
        """
        batch_size = state.size(0)
        
        # Codifica gli asset
        encoded_state = self.asset_encoder(state, self.action_size)
        
        # Applica attenzione se abilitata
        if self.use_attention:
            encoded_state = self.apply_attention(encoded_state, batch_size)
        
        # Inizializza dinamicamente i layer FC se necessario
        if not self.layers_initialized or self.fc1.weight.shape[1] != encoded_state.size(1):
            self.initialize_layers(encoded_state.size(1))
        
        # Feed-forward con Layer Normalization (funziona anche con batch_size=1)
        x = F.relu(self.ln1(self.fc1(encoded_state)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        # Output layer
        return self.fc3(x)