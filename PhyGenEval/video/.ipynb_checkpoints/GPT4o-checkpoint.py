import cv2
# from moviepy.editor import VideoFileClip
import time
import base64
import os
import json
import base64
import requests
# import jsonlines
from multiprocessing import Pool
from openai import OpenAI
from functools import partial
import time
# from IPython.display import Image, display, Audio, Markdown

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key='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',
)

# We'll be using the OpenAI DevDay Keynote Recap video

def process_video(video_path, num_frames_to_sample):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the seconds per frame needed to sample the desired number of frames
    video_duration_seconds = total_frames / fps
    seconds_per_frame = video_duration_seconds / num_frames_to_sample
    frames_to_skip = int(fps * seconds_per_frame)
    
    curr_frame = 0

    # Loop through the video and extract frames at the calculated sampling rate
    while curr_frame < total_frames - 1 and len(base64Frames) < num_frames_to_sample:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    print(f"{video_path}: Extracted {len(base64Frames)} frames")
    return base64Frames

modellist = ['phygen_r20']
with open('/workspace/PhyGenBench-Test/PhyGenBench/video_question.json','r') as f:
    data = json.load(f)

for modelname in modellist:
    # ====== 新增：断点续跑逻辑 ======
    save_path = f'/workspace/PhyGenBench-Test/PhyGenEval/video/prompt_replace_augment_video_question_{modelname}_res.json'
    
    start_idx = 0
    if os.path.exists(save_path):
        # 已经有完整 json，直接读出来，继续在后面接着跑
        with open(save_path, 'r') as f:
            old_result = json.load(f)
        print("Load previous result, len =", len(old_result))
        result = old_result
        start_idx = len(old_result)
    else:
        result = []
        print("No previous result, start from 0.")
    
    # =========================
    
    for i in range(start_idx, len(data)):
        data_tmp = data[i]
    
        prompt = """### Task Overview:
    
        Your task is to analyze an input video to determine whether it conforms to real-world physical laws. You will receive the T2V prompt corresponding to this video, as well as the physical law it primarily reflects. Besides, you will be provided with four different descriptions (Completely Fantastical, Highly Unrealistic, Slightly Unrealistic, Almost Realistic) that offer varying levels of detail or focus. Your goal is to select the most appropriate description to evaluate the extent to which this video conforms to the emphasized physical law.
    
        ### Task Requirements:
    
        1. **Selection**: Choose the description that best suits the purpose of assessing the video’s physical realism.
        2. **Explanation**: Provide a reason for your selection, explaining why this description is the most relevant for the task.
    
        ### Expected Output Format:
    
        {
        "Choice": "<Selected_Description>",
        "Reason": "<Explanation>"
        }
    
        ### Special Notes:
        
        - Exercise caution when assigning choices, especially when considering the Almost Realistic.
        - Do not easily give the choice of Almost Realistic.
        - Use step-by-step reasoning to make your selection, considering the relevance and specificity of each description.
        - The explanation should be concise but comprehensive, highlighting key factors that influenced your choice.
        - You need to focus on whether the video reflects the emphasized physical law.
    
        """
    
        video_path = f'/workspace/{modelname}/video_{i+1}.mp4'
        print(f"[INFO] index {i}, video_path = {video_path}")
    
        base64Frames = process_video(video_path, num_frames_to_sample=25)
    
        input_prompt = f"""
        Here is the t2V prompt and the physical law it primarily reflects:
        Prompt:{data_tmp['caption']}
        Physical_Law:{data_tmp['physical_laws']}
        Here is the different descriptions:
        Completely Fantastical:{data_tmp['video_question']['Description1']}
        Highly Unrealistic:{data_tmp['video_question']['Description2']}
        Slightly Unrealistic:{data_tmp['video_question']['Description3']}
        Almost Realistic:{data_tmp['video_question']['Description4']}
        """
    
        full_prompt = prompt + input_prompt
    
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Always use double quotes for keys and values, and never use single quotes or any extra text."
                    },
                    {"role":"user","content":full_prompt},
                    {"role":"user", "content":[
                        "These are the frames from the video.",
                        *map(lambda x:{"type":"image_url",
                                        "image_url":{"url":f'data:image/jpg;base64,{x}',"detail":"low"}}, base64Frames)
                        ],
                    }
                ],
                temperature=0,
                max_tokens=400,
            )
    
            raw_response = response.choices[0].message.content
    
            if raw_response is None or not raw_response.strip():
                print(f"[WARNING] Empty response for index {i}.")
                parsed_data = {"raw_content": ""}
            else:
                start = raw_response.find("{")
                end = raw_response.rfind("}") + 1
                json_block = raw_response[start:end].strip() if start != -1 and end != -1 else ""
    
                try:
                    parsed_data = json.loads(json_block)
                except (json.JSONDecodeError, ValueError):
                    print(f"[WARNING] JSON parse failed at index {i}, store raw content.")
                    parsed_data = {"raw_content": raw_response.strip()}
    
        except Exception as e:
            print(f"[ERROR] API error at index {i}: {e}")
            parsed_data = {"raw_content": f"API error: {str(e)}"}
    
        print(parsed_data)
    
        data_tmp['GPT4o'] = parsed_data
    
        # score 映射
        if (
            list(parsed_data.keys()) == ['raw_content']
            and parsed_data['raw_content'].strip().startswith("I'm unable to analyze or interpret images directly")
        ):
            score = 0
        else:
            choice = parsed_data.get("Choice")
            if choice is None:
                print(f"[WARNING] No 'Choice' field found in response at index {i}. Using score = 0.")
                score = 0
            else:
                if 'Completely Fantastical' in choice:
                    score = 0
                elif 'Highly Unrealistic' in choice:
                    score = 1
                elif 'Slightly Unrealistic' in choice:
                    score = 2
                elif 'Almost Realistic' in choice:
                    score = 3
                else:
                    print(f"[WARNING] Unrecognized choice value at index {i}: {choice}. Using score = 0.")
                    score = 0
    
        data_tmp['GPT4o_score'] = score
    
        # 更新 result 列表
        result.append(data_tmp)
    
        # ✅ 每处理一个样本就保存一次，方便中断后恢复
        with open(save_path, 'w') as f:
            json.dump(result, f)
    
        print(f"[INFO] Saved up to index {i}, current result len = {len(result)}")
    
    
    print("[INFO] Done. Total result len =", len(result))
    # 再保存一遍最终版（可选）
    with open(save_path, 'w') as f:
        json.dump(result, f)
    


