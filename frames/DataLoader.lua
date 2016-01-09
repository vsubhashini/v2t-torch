local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)

  local info = path.join(opt.save_dir, 'info.json')

  local prepro = false
  if not path.exists(info) then
    prepro = true
  else
    local curDataInfo = utils.read_json(info)
    if curDataInfo['batch_size'] ~= opt.batch_size or curDataInfo['max_seqlen'] ~= opt.max_seqlen then
      prepro = true
    end
  end

  if prepro then
    print('running one time pre-processing...')
    local labelData = {path.join(opt.label_dir, 'labels_train.txt'), path.join(opt.label_dir, 'labels_val.txt')}
    local videoData = {path.join(opt.video_dir, 'videos_train.txt'), path.join(opt.video_dir, 'videos_val.txt')}

    self:preprocess(labelData, videoData, opt.vocab_file, opt.save_dir, opt.batch_size, opt.max_seqlen)
  end

  print('loading preprocessed files...')
  local dataInfo = utils.read_json(info)

  -- load vocabulary
  self.word_to_ix = dataInfo['word_to_ix']
  self.ix_to_word = dataInfo['ix_to_word']
  self.vocabSize = 0
  for k,v in pairs(self.word_to_ix) do
    self.vocabSize = self.vocabSize + 1
  end 

  self.save_dir = opt.save_dir
  self.iter = {0, 0, 0}
  self.splitNames = {'train', 'val', 'valEval'}
  self.splitSizes = dataInfo['splitSizes']
end

function DataLoader:getBatch(split_ix)
  self.iter[split_ix] = self.iter[split_ix]+1
  if self.iter[split_ix] > self.splitSizes[split_ix] then
    self.iter[split_ix] = 1
  end

  local batchFile = self.splitNames[split_ix] .. self.iter[split_ix] .. '.t7'
  local batch = torch.load(path.join(self.save_dir, batchFile))
  return batch[1], batch[2], batch[3]
end

function DataLoader:getVocabSize()
  return self.vocabSize
end

function DataLoader:getVocab()
  return self.ix_to_word
end

function DataLoader:splitSize(split_ix)
  return self.splitSizes[split_ix]
end

function DataLoader:resetIterator(split_ix)
  self.iterators[split_ix] = 0
end

function DataLoader:preprocess(label_files, video_files, vocab_file, save_dir, batch_size, max_seqlen)
  local rawdata, rawdata2 -- two streams for reading raw data

  local out = {} -- info json struct 
  out['batch_size'] = batch_size
  out['max_seqlen'] = max_seqlen

  print('creating vocabulary mapping...')
  local word_to_ix = {}
  local ix_to_word = {}

  local f = io.open(vocab_file)
  rawdata = f:read()
  local vocabSize = 0
  repeat 
    vocabSize = vocabSize + 1
    word_to_ix[rawdata] = vocabSize
    ix_to_word[vocabSize] = rawdata
    rawdata = f:read()
  until not rawdata
  f:close()

  out['word_to_ix'] = word_to_ix
  out['ix_to_word'] = ix_to_word

  local splits = {'train', 'val'}
  local splitSizes = {}

  for splitIx=1, #splits do

    print('processing labels and videos for ' .. splits[splitIx] .. '...')
    local vidlabelPairs = {} -- table that stores video-caption pairs 
    local ids = {} -- store ids that correspond to entries in vidlabelPairs 
    local labelFile = io.open(label_files[splitIx])
    local vidFile = io.open(video_files[splitIx])

    rawdata = labelFile:read()
    rawdata2 = vidFile:read()
    local curCapId = utils.split(rawdata, '%s')[1]
    local curVidCaps = {}
    local longestCap = 0

    repeat
      local curCap = utils.split(rawdata, '%s')
      local curCapLen = #curCap
      if curCapLen > longestCap then longestCap = curCapLen end
      local capTensor = torch.Tensor(curCapLen):zero() -- #sent because sent[1] is the video id but add END token to tensor
      for ix=2, curCapLen do
        capTensor[ix-1] = word_to_ix[curCap[ix]]
      end
      capTensor[curCapLen] = vocabSize + 1 -- END TOKEN
      table.insert(curVidCaps, capTensor)

      rawdata = labelFile:read()
      if rawdata == nil or utils.split(rawdata, '%s')[1] ~= curCapId then -- read the video
        local curVid = {}
        local maxFrames = max_seqlen - longestCap
        local frameCount = 0
        repeat
          frameCount = frameCount + 1
          local convnetFeats = utils.split(rawdata2, ',')
          local frame = torch.Tensor(#convnetFeats-1)
          for ix=2,#convnetFeats do
            frame[ix-1] = convnetFeats[ix]
          end
          table.insert(curVid, frame)
          rawdata2 = vidFile:read()
          if rawdata2 == nil or utils.split(utils.split(rawdata2, ',')[1], '_')[1] ~= curCapId or frameCount == max_frames then
            if frameCount == maxFrames then
              repeat
                rawdata2 = vidFile:read()
              until not rawdata2 or utils.split(utils.split(rawdata2, ',')[1], '_')[1] ~= curCapId
            end
            break
          end
        until not rawdata2

        for ix=1,#curVidCaps do
          table.insert(vidlabelPairs, {curVid, curVidCaps[ix]})
          table.insert(ids, curCapId)
        end
        curVidCaps = {}
        longestCap = 0
        if rawdata ~= nil then curCapId = utils.split(rawdata, '%s')[1] end
      end
    until not rawdata

    collectgarbage()

    print('generating batches for ' .. splits[splitIx] .. '...')
    -- create batches
    local batches = {}
    -- make even bathes
    if #vidlabelPairs % batch_size ~=0 then 
      repeat
        table.remove(vidlabelPairs, #vidlabelPairs)
      until #vidlabelPairs % batch_size == 0
    end

    local splitSize = #vidlabelPairs/batch_size
    table.insert(splitSizes, splitSize)
    local rand = torch.randperm(#vidlabelPairs)
    for b=1,splitSize do
      local longestCap = 0
      local longestVid = 0
      -- find longest video and longest sentence 
      for b_ix=1,batch_size do
        local ix = rand[(b-1)*batch_size + b_ix]
        if #vidlabelPairs[ix][1] > longestVid then
          longestVid = #vidlabelPairs[ix][1]
        end

        if vidlabelPairs[ix][2]:nElement() > longestCap then
          longestCap = vidlabelPairs[ix][2]:nElement()
        end
      end

      if longestVid + longestCap > max_seqlen then
        longestVid = max_seqlen - longestCap
      end

      local batchVideos = {}
      for i=1,longestVid do table.insert(batchVideos, torch.Tensor(batch_size, 4096):zero()) end
      local batchLabels = torch.Tensor(longestCap, batch_size):zero()
      local batchIds = {}
      for b_ix=1,batch_size do
        local ix = rand[(b-1)*batch_size + b_ix]
        local cap = vidlabelPairs[ix][2]
        batchLabels:select(2,b_ix):sub(1,cap:nElement()):copy(cap)

        local vid = vidlabelPairs[ix][1]
        local fill
        if #vid < longestVid then fill = #vid else fill = longestVid end
        for frame_num=1,fill do
          batchVideos[frame_num]:select(1,b_ix):copy(vid[frame_num])
        end
        table.insert(batchIds, ids[ix])
      end
      torch.save(path.join(save_dir, splits[splitIx] .. b .. '.t7'), {batchVideos, batchLabels, batchIds})
    end

    if splitIx == 2 then
      print('generating batches for valEval...')
      local valEval = {}
      local idDict = {}
      local count = 0
      for s=1,#vidlabelPairs do
        if idDict[ids[s]] == nil then
          count = count + 1
          table.insert(valEval, {vidlabelPairs[s][1], vidlabelPairs[s][2], ids[s]})
          idDict[ids[s]] = true
          torch.save(path.join(save_dir, 'valEval' .. count .. '.t7'), {vidlabelPairs[s][1], vidlabelPairs[s][2], ids[s]})
        end
      end
      table.insert(splitSizes, #valEval)
    end

    collectgarbage()
  end

  out['splitSizes'] = splitSizes
  utils.write_json(path.join(save_dir, 'info.json'), out)
end