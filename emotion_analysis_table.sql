-- Create emotion_analysis table to store assessment results
CREATE TABLE IF NOT EXISTS emotion_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Overall emotion analysis
    overall_emotion TEXT NOT NULL,
    overall_confidence NUMERIC NOT NULL,
    
    -- Question details stored as JSON arrays
    questions JSONB NOT NULL,
    
    -- Additional metadata
    device_info TEXT,
    session_duration INTEGER,
    
    -- Support for querying
    UNIQUE(user_id, session_id)
);

-- Enable RLS (Row Level Security)
ALTER TABLE emotion_analysis ENABLE ROW LEVEL SECURITY;

-- Create policies for secure access
-- Users can view their own emotion analysis results
CREATE POLICY "Users can view their own emotion analysis" ON emotion_analysis
    FOR SELECT USING (auth.uid() = user_id);

-- Only the app server can insert new records
CREATE POLICY "App server can insert emotion analysis" ON emotion_analysis
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users cannot update their emotion analysis results
CREATE POLICY "Users cannot update emotion analysis" ON emotion_analysis
    FOR UPDATE USING (false);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_emotion_analysis_user_id ON emotion_analysis (user_id);
CREATE INDEX IF NOT EXISTS idx_emotion_analysis_created_at ON emotion_analysis (created_at);
CREATE INDEX IF NOT EXISTS idx_emotion_analysis_overall_emotion ON emotion_analysis (overall_emotion);

-- Create a JSONB index for the questions field to speed up JSON queries
CREATE INDEX IF NOT EXISTS idx_emotion_analysis_questions ON emotion_analysis USING GIN (questions);

-- Sample function to retrieve aggregate emotion stats for a user
CREATE OR REPLACE FUNCTION get_user_emotion_stats(p_user_id UUID)
RETURNS TABLE (
    emotion TEXT,
    count BIGINT,
    average_confidence NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        overall_emotion as emotion,
        COUNT(*) as count,
        AVG(overall_confidence) as average_confidence
    FROM 
        emotion_analysis
    WHERE 
        user_id = p_user_id
    GROUP BY 
        overall_emotion
    ORDER BY 
        count DESC;
END;
$$ LANGUAGE plpgsql; 