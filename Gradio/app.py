from typing import List, Tuple
import gradio as gr
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Categories in English and Maltese
categories_english = {
    0: 'Jesus has Risen', 1: 'Jesus praying in Gethsemane', 2: 'Saint Philip of Agira',
    3: 'Simon of Cyrene', 4: 'The Betrayal of Judas', 5: 'The Cross',
    6: 'The Crucifixion', 7: 'The Ecce Homo', 8: 'The Flogged',
    9: 'The Lady of Sorrows', 10: 'The Last Supper', 11: 'The Monument',
    12: 'The Redeemer', 13: 'The Veronica'
}

categories_maltese = {
    0: 'L-Irxoxt', 1: 'Ġesù fl-Ort tal-Ġetsemani', 2: 'San Filep ta’ Aġġira',
    3: 'Xmun min Ċireni', 4: 'It-Tradiment ta’ Ġuda', 5: 'Is-Salib',
    6: 'Il-Vara L-Kbira', 7: 'L-Ecce Homo', 8: 'Il-Marbut',
    9: 'Id-Duluri', 10: 'L-Aħħar Ċena', 11: 'Il-Monument',
    12: 'Ir-Redentur', 13: 'Il-Veronica'
}

categories_synopsis = {
    0: "The 'Jesus Has Risen' story recounts Jesus Christ's resurrection on the third day after His crucifixion. Women visiting His tomb find it empty, and an angel announces He has risen. Jesus appears to Mary Magdalene, His disciples, and others, confirming His victory over death and fulfilling prophecies. This event signifies hope, renewal, and eternal life for believers, forming the foundation of Christian faith and the celebration of Easter.\n\n"
       "L-Istorja ta’ 'L-Irxoxt' tirrakkonta l-qawmien ta’ Ġesù Kristu fit-tielet jum wara s-salib tiegħu. Nisa li żaru l-qabar tiegħu sabuh vojt, u anġlu ħabbar li qam. Ġesù deher lil Marija Maddalena, lid-dixxipli tiegħu, u lil oħrajn, jikkonferma r-rebħa tiegħu fuq il-mewt u jwettaq il-profeziji. Dan l-avveniment ifisser tama, tiġdid, u ħajja eterna għall-fidili, u jifforma l-bażi tal-fidi Nisranija u ċ-ċelebrazzjoni tal-Għid.",

    1: "The 'Jesus Praying in Gethsemane' scene occurs after the Last Supper. Jesus retreats to the Garden of Gethsemane with His disciples to pray. Overwhelmed by the weight of His impending crucifixion, He pleads with God to let the cup of suffering pass from Him but submits to God's will. This moment reflects Jesus' humanity and divine mission, highlighting His obedience and deep sorrow.\n\n"
       "Ix-xena ta’ 'Ġesù fl-Ort tal-Ġetsemani' sseħħ wara L-Aħħar Ċena. Ġesù mar lura fil-Ġnien tal-Ġetsemani mad-dixxipli tiegħu biex jitlob. Maħkum mill-piż tas-salib li kien ġej, talab lil Alla biex il-kalċi tas-sofferenza jgħaddi minn fuqu iżda ssottometta ruħu għar-rieda ta’ Alla. Dan il-mument jirrifletti l-umanità u l-missjoni divina ta’ Ġesù, u emfasizza l-ubbidjenza u l-uġigħ profond tiegħu.",

    2: "Saint Philip of Agira was an early Christian exorcist and preacher, celebrated for his faith and miracles. Born in Antioch, Philip evangelised Sicily, converting many to Christianity. He is often depicted in religious art exorcising demons and symbolising triumph over evil. His story inspires faith and devotion to God’s power working through saints.\n\n"
       "San Filep ta’ Aġġira kien eżorċista u predikatur Nisrani bikri, iċċelebrat għall-fidi u l-mirakli tiegħu. Imwieled f’Antjokja, Filep evangelizza lil Sqallija, u kkonverta lil ħafna għall-Kristjaneżmu. Ħafna drabi jidher fl-arti reliġjuża jeżorċizza x-xjaten u jissimbolizza r-rebħa fuq il-ħażen. L-istorja tiegħu tispira fidi u devozzjoni lejn il-qawwa ta’ Alla li taħdem permezz tal-qaddisin.",

    3: "Simon of Cyrene was compelled to help Jesus carry His cross on the way to Calvary. This act, though forced, became a symbol of discipleship, compassion, and sharing in Christ's suffering. Simon’s inclusion in the Passion narrative shows the importance of serving others and bearing one another's burdens.\n\n"
       "Xmun minn Ċireni ġie mġiegħel jgħin lil Ġesù jerfa’ s-salib tiegħu fit-triq lejn il-Kalvarju. Dan l-att, għalkemm imġiegħel, sar simbolu ta’ dixxipulat, kompassjoni, u qsim fit-tbatija ta’ Kristu. L-istorja ta’ Xmun tenfasizza l-importanza li ngħinu lil xulxin u nerfgħu l-piżijiet ta’ xulxin.",

    4: "The 'Betrayal of Judas' recounts Judas Iscariot’s betrayal of Jesus to the authorities for thirty pieces of silver. Judas identifies Jesus with a kiss in Gethsemane, leading to His arrest. This act represents greed, treachery, and the fulfilment of prophecy, contrasting with Jesus' forgiveness and sacrifice.\n\n"
       "It-Tradiment ta’ Ġuda jirrakkonta kif Ġuda Iskarjota trabba lil Ġesù għand l-awtoritajiet għal tletin biċċa fidda. Ġuda identifika lil Ġesù b’bewsa fl-Ort tal-Ġetsemani, li wasslet għall-arrest tiegħu. Dan l-att jirrappreżenta l-għatx għall-flus, il-qerq, u t-twettiq tal-profeziji, li jikkuntrasta ma’ l-maħfra u s-sagrifiċċju ta’ Ġesù.",

    5: "The Cross symbolises Jesus' crucifixion and His ultimate sacrifice for humanity's redemption. It represents suffering, forgiveness, and victory over sin. Christians venerate the Cross as a sign of salvation, reflecting on the love and grace of Jesus’ sacrifice to reconcile humanity with God.\n\n"
       "Is-Salib jissimbolizza l-kruċifissjoni ta’ Ġesù u s-sagrifiċċju finali tiegħu għall-iskop tal-fidwa tal-umanità. Jirrappreżenta sofferenza, maħfra, u rebħa fuq il-ħtija. Il-Kristjani jiveneraw is-Salib bħala sinjal tas-salvazzjoni, jirriflettu dwar l-imħabba u l-grazzja tas-sagrifiċċju ta’ Ġesù biex jirrikonċilja l-umanità ma’ Alla.",

    6: "The 'Crucifixion' describes Jesus' execution at Golgotha. Nailed to the Cross, He endures physical agony and mockery from onlookers. Despite His suffering, Jesus offers forgiveness to His persecutors and entrusts His spirit to God. The Crucifixion signifies the ultimate act of love and redemption for humanity’s sins.\n\n"
       "Il-Vara L-Kbira tirrakkonta l-eżekuzzjoni ta’ Ġesù fil-Golgota. Maħkum bil-pali, Ġesù sofra agunija fiżika u żmerċ ta’ dawk li kienu qed jarawh. Minkejja s-sofferenza tiegħu, Ġesù offra maħfra lil dawk li kienu qed jippersegwitawh u ta l-ispirtu tiegħu lil Alla. Il-Vara L-Kbira tissimbolizza l-aħħar att ta’ imħabba u fidwa mill-ħtija tal-umanità.",

    7: "The 'Ecce Homo' moment occurs when Pontius Pilate presents Jesus to the crowd after His scourging, saying, 'Behold the man!' This scene highlights Jesus’ humility, suffering, and the rejection He faced. It emphasises His innocence amidst unjust condemnation, leading to His crucifixion.\n\n"
       "L-Ecce Homo jirrakkonta l-mument meta Pontju Pilatu ippreżenta lil Ġesù lill-folla wara li ġie msajjar, u qal: 'Ara l-bniedem!' Din ix-xena tinfoka fuq l-umiltà ta’ Ġesù, is-sofferenza tiegħu, u r-rifjut li sofra. Tikkonferma l-innoċenza tiegħu filwaqt li kien inqabad b’mod inġust, li wassal għall-kruċifissjoni tiegħu.",

    8: "The 'Flogged' refers to Jesus' scourging before His crucifixion. He is brutally whipped by Roman soldiers, enduring immense pain and humiliation. This moment reflects Jesus’ willing acceptance of suffering for humanity’s redemption, demonstrating His steadfast love and sacrifice for sinners.\n\n"
       "Il-Marbut jirrakkonta l-kastig ta’ Ġesù qabel il-kruċifissjoni tiegħu. Ġesù ġie fustigat b’mod brutali mill-suldati Rumani, u sofra minn uġigħ u umiljazzjoni kbira. Dan il-mument jirrifletti l-acceptazzjoni volontarja ta’ Ġesù tas-sofferenza għad-dinja, u juri l-imħabba u s-sagrifiċċju tiegħu għas-sinners.",

    9: "The 'Lady of Sorrows' focuses on Mary, the mother of Jesus, and her deep suffering during His Passion and death. Her grief, often depicted in art with seven symbolic swords piercing her heart, represents her unwavering faith and love. She becomes a model of compassion and endurance for believers.\n\n"
       "Id-Duluri tenfasizza l-istorja ta’ Marija, omm Ġesù, u s-sofferenza profonda tagħha waqt il-Passjoni u l-mewt ta’ Ġesù. L-uġigħ tagħha, spiss irrapreżentata fl-arti bi seba’ sriedaq simboliċi li jtaqqbu qalbha, tirrappreżenta l-fidi u l-imħabba sħiħa tagħha. Marija ssir mudell ta’ kompassjoni u sodisfazzjon għall-fidili.",
    
    10: "The 'Last Supper' depicts Jesus' final meal with His disciples before His arrest. During the meal, He institutes the Eucharist, offering bread and wine as His body and blood. The Last Supper signifies Jesus' love, sacrifice, and the establishment of a new covenant with humanity.\n\n"
        "L-Aħħar Ċena tirrakkonta l-ikel finali ta’ Ġesù mad-dixxipli tiegħu qabel l-arrest tiegħu. Matul il-ikel, Ġesù jistabbilixxi l-Ewkaristija, fejn joffri ħobż u inbid bħala ġismu u demmu. L-Aħħar Ċena tirrifletti l-imħabba ta’ Ġesù, is-sagrifiċċju tiegħu, u l-istabbiliment tal-patt il-ġdid mal-umanità.",

    11: "The 'Monument' refers to the solemn tomb where Jesus' body was laid after His crucifixion. Guarded and sealed, the tomb signifies the end of Jesus’ earthly mission and the waiting period before His resurrection. It becomes a symbol of hope and triumph over death.\n\n"
        "Il-Monument jirreferi għall-qabar solenni fejn tqiegħed il-ġisem ta’ Ġesù wara l-kruċifissjoni tiegħu. Miżmum u sigillat, il-qabar jissimbolizza t-tmiem tal-missjoni terrena ta’ Ġesù u l-perjodu ta’ stennija qabel ir-risurrezzjoni tiegħu. Dan isir simbolu ta’ tama u rebħa fuq il-mewt.",

    12: "The 'Redeemer' emphasises Jesus as the Saviour who reconciles humanity with God through His sacrifice. His death and resurrection embody love, mercy, and victory over sin. The Redeemer serves as the cornerstone of Christian faith, offering eternal life to all who believe in Him.\n\n"
        "Ir-Redentur jenfasizza lil Ġesù bħala s-Salvatur li jirrikonċilja l-umanità ma’ Alla permezz tas-sagrifiċċju tiegħu. Il-mewt u r-risurrezzjoni tiegħu jinkorporaw l-imħabba, il-ħniena, u r-rebħa fuq id-dnub. Ir-Redentur iservi bħala l-ġebla tax-xewka tal-fidi Nisranija, u joffri ħajja eterna lil dawk kollha li jemmnu fih.",

    13: "The 'Veronica' story recounts a woman who compassionately wiped Jesus’ face with a cloth during His journey to Calvary. The cloth miraculously retained His image. This act represents compassion and faith, reminding Christians of the call to show love and mercy in the face of suffering.\n\n"
        "Il-Veronika tirrakkonta l-istorja ta’ mara li b’kompassjoni xħanqet wiċċ Ġesù b’biċċa drapp waqt il-vjaġġ tiegħu lejn il-Kalvarju. Il-drapp żamm l-immaġni ta’ Ġesù b’mod mirakoluż. Dan l-att jirrappreżenta kompassjoni u fidi, u jfakkar lill-Kristjani fl-obbligu li juru imħabba u ħniena quddiem it-tbatija."
}


def predict_image(image, model, size=(244, 244)) -> List[Tuple[str, str, float]]:
    """Predict the class of a given image and return sorted probabilities with categories."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(image, size)
    resized_img = resized_img / 255.0  # Normalize
    resized_img = resized_img.transpose(2, 0, 1)  # Convert to (C, H, W)
    resized_img = resized_img[None, ...]  # Add batch dimension

    # Run prediction
    results = model.predict(image)
    pred_probs = results[0].probs.data.cpu().numpy()

    # Sort predictions by probability
    sorted_indices = np.argsort(pred_probs)[::-1]  # Descending order
    sorted_predictions = [
        (
            categories_english[i],
            categories_maltese[i],
            round(pred_probs[i] * 100, 2)  # Convert to percentage
        )
        for i in sorted_indices
    ]

    return sorted_predictions

# Load the model
model = YOLO("https://huggingface.co/mbar0075/Maltese-Christian-Statue-Classification/resolve/main/MCS-Classify.pt").to(device)

example_list = [["examples/" + example] for example in os.listdir("examples")]

def classify_image(input_image):
    # Get predictions from the model
    predictions = predict_image(input_image, model)
    
    # Format predictions into a dictionary with confidence scores
    formatted_predictions = {
        f"{label} / {maltese_label}": confidence / 100
        for label, maltese_label, confidence in predictions[:5]
    }
    
    # Get the label with the highest confidence
    highest_confidence_label = predictions[0][0]  # Assuming predictions are sorted by confidence

    # Get the corresponding key for the label
    highest_confidence_label = list(categories_english.keys())[list(categories_english.values()).index(highest_confidence_label)]
    highest_confidence_synopsis = categories_synopsis.get(highest_confidence_label)
    
    # Return the formatted predictions, the highest-confidence synopsis, and FPS
    return (
        formatted_predictions,
        highest_confidence_synopsis,
        round(1.0 / 0.1, 2)  # Example FPS calculation
    )

# Metadata
title = "Maltese Christian Statue Image Classification ✝"
description = (
    "This project aims to classify Maltese Christian statues and religious figures depicted in images. The model is trained "
    "to recognise 14 unique categories related to the Maltese Christian heritage, presented in both English and Maltese. "
    "Upload an image to receive predictions of the statue's category, sorted by likelihood. This initiative promotes the "
    "preservation of Maltese culture and traditions through AI technology."
)
article = (
    "The YOLO11m classification model was trained on a dataset of Maltese Christian statues and religious figures during the period of Lent.\n"
    "The MCS Dataset is open-source and available for access through https://github.com/mbar0075/Maltese-Christian-Statue-Classifier.\n"
    "\n © Matthias Bartolo 2025. Licensed under the MIT License."
)

# Create the Gradio demo
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions (English / Maltese)"),
        gr.Textbox(label="Synopsis / Aktar Tagħrif"),
        gr.Number(label="Prediction speed (FPS)")
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

# Launch the demo
demo.launch()
